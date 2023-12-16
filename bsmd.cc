/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 */

// @sect3{Include files}
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_stack.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/tensor.h>

// This is needed for C++ output:
#include <iostream>
#include <fstream>
// And this for the declarations of the `std::sqrt` and `std::fabs` functions:
#include <cmath>
// For runtime calculations.
#include <chrono>
// For the random number generation.
#include <random>

using namespace dealii;



// ############### Begin setting parameters ###################
  constexpr unsigned int dimension = 2;
  const unsigned int n_time_steps = 50;
  const double maturity_time = 1.0;
  double time_step = maturity_time / n_time_steps;
  const double maximum_stock_price = 5.0;
  constexpr int tensor_dim = 3;
  const Tensor<2, tensor_dim> rho({ {1,.5,.6}
                                    ,{.5,1,.7}
                                    ,{.6,.7,1}});
  const Tensor<1, tensor_dim> sigma( {.2,.7,.3} );
  const double interest_rate = .05;

  // Uncomment this to enable the Method of Manufactured Solutions (MMS).
  // #define MMS

  // Uncomment this to allow for random noise to be added.
  // #define RAND

  #ifndef MMS
    const double strike_price = 3;
    constexpr unsigned int initial_global_refinement = 5;
  #else
    constexpr unsigned int initial_global_refinement = 0;
  #endif

  #ifdef RAND
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10,10);
  #endif
// ############### End setting parameters ###################

/**
 * Prints a point on the screen.
 * 
 * This is a helper function that will print the contents of a point to the console.
 * 
 * @param p The point to be printed to the console.
*/
template <int dim>
void printPoint(const Point<dim> &p)
{
  std::cout << "[";
  for (int i = 0; i < dim; i++)
  {
    std::cout << p[i];
    if (i != dim - 1)
    {
      std::cout << " ";
    }
  }
  std::cout << "]" << std::endl;
}

// ## Helper Methods ##

/**
 * Implements the choose function. 
 * 
 * This returns the number of different ways that k objects can be chosen from n objects.
 * 
 * @param n The size of the set of objects being chosen from.
 * @param k The number of elemtents to choose from the set.
 * 
 * @returns The number of combinations that can be formed when k objects are chsosen from a set of n.
*/
uint64_t choose(uint64_t n, uint64_t k)
{
  if (k == 0)
  {
    return 1;
  }
  return (n * choose(n-1, k-1) / k);
}

/**
 * Returns the boundary ID.
 * 
 * NEED TO DOCUMENT THIS FURTHER.
*/
uint32_t boundary_id(const uint64_t missing_axis, const uint64_t finalDim)
{
  uint32_t ret = 1;
  if (finalDim != 1)
  {
    uint32_t full_bits = (1 << finalDim) - 1;
    ret = full_bits ^ (1 << missing_axis);
  }

  return ret;
}

template <int dim>
std::array<uint32_t, dim> get_axis_ids(uint32_t boundary_id)
{
  std::array<uint32_t, dim> axis_ids;
  axis_ids.fill(0);

  uint32_t bits_found = 0;
  uint32_t counter = 0;
  while ( (bits_found < dim) && (counter < 32) ) // Because 32 bits
  {
    if (boundary_id & 1)
    {
      axis_ids[bits_found] = counter;
      bits_found++;
    }
    boundary_id >>= 1;
    counter++;
  }
  assert(bits_found == dim);

  return axis_ids;
}

template <int dim, int TensorDim>
Tensor<2,dim> create_rho_matrix(uint32_t b_id, const Tensor<2,TensorDim> rho_matrix)
{
  auto axis_ids = get_axis_ids<dim>(b_id);
  Tensor<2,dim> ret_matrix;

  for (uint32_t row = 0; row < TensorDim; row++)
  {
    for (uint32_t col = 0; col < TensorDim; col++)
    {
      auto rowIdx = std::find(axis_ids.begin(), axis_ids.end(), row) - axis_ids.begin();
      auto colIdx = std::find(axis_ids.begin(), axis_ids.end(), col) - axis_ids.begin();

      if ( (rowIdx != dim) && (colIdx != dim))
      {
        ret_matrix[rowIdx][colIdx] = rho_matrix[row][col];
      }
    }
  }

  return ret_matrix;
}

template <int dim, int TensorDim>
Tensor<1,dim> create_sigma_vector(const uint32_t b_id, const Tensor<1,TensorDim> sigma_vector)
{
  auto axis_ids = get_axis_ids<dim>(b_id);
  Tensor<1,dim> ret_vector;

  for (uint32_t sigmaIdx = 0; sigmaIdx < TensorDim; sigmaIdx++)
  {
    auto vecIdx = std::find(axis_ids.begin(), axis_ids.end(), sigmaIdx) - axis_ids.begin();

    if (vecIdx != dim)
    {
      ret_vector[vecIdx] = sigma_vector[sigmaIdx];
    }
  }

  return ret_vector;
}

// ### MMS Solution ###
#ifdef MMS

/**
 * Solution Class
 * 
 * This class is used for validating the program when using MMS. The function being chosen here
 * is the inverted parabola moved up 6 units.
*/
template <int dim>
class Solution: public Function<dim>
{
  // -t^2 - S_i^2 + 6
public:
  Solution();
  virtual double value(const Point<dim> &p,
                      const unsigned int component = 0) const override;

  virtual Tensor<1,dim>
  gradient(const Point<dim> &p,
          const unsigned int component = 0) const override;
};

// Default Constructor
template <int dim>
Solution<dim>::Solution()
{}

/**
 * Returns the value of the solution evaluated at a given point.
 * 
 * Here, component is unused, but is necessary as an argument because of the inheritance.
 * 
 * @param p The point to evaluate at.
 * @param component An unused parameter.
 * 
 * @returns The value of the solution at the given point.
*/
template <int dim>
double Solution<dim>::value(const Point<dim> &p,
                      const unsigned int component) const
{
  (void) component;

  double ret = -Utilities::fixed_power<2,double>(this->get_time()) + 6;
  for (unsigned int i = 0; i < dim; i++)
  {
    ret += -Utilities::fixed_power<2,double>(p(i));
  }

  return ret;
}

template <int dim>
Tensor<1,dim> Solution<dim>::gradient(const Point<dim> &p,
                                const unsigned int component) const
{
  (void) component;

  Tensor<1,dim> ret;
  for (unsigned int i = 0; i < dim; i++)
  {
    ret[i] = -2 * p(i);
  }

  return ret;
}

#endif
// ## End of MMS Solution ##



// ## A Matrix ##
template <int dim>
class AMatrix: public TensorFunction<2,dim>
{
public:
  AMatrix(const Tensor<1,dim> sigmaTT,
          const Tensor<2,dim> rhoTT);

  virtual Tensor<2,dim>
  value(const Point<dim> &p) const override;

  Tensor<1,dim>
  divergence(const Point<dim> &p) const;

private:
  Tensor<1,dim> _sigma;
  Tensor<2,dim> _rho;
};

// Constructor
template <int dim>
AMatrix<dim>::AMatrix(const Tensor<1,dim> sigmaTT,
                      const Tensor<2,dim> rhoTT)
  : TensorFunction<2,dim>()
  , _sigma(sigmaTT)
  , _rho(rhoTT)
{}

template <int dim>
Tensor<2,dim>
AMatrix<dim>::value(const Point<dim> &p) const
{
  Tensor<2,dim> a_matrix;

  for (unsigned int i = 0; i < dim; i++)
  {
    for (unsigned int j = 0; j < dim; j++)
    {
      a_matrix[i][j] = 0.5 * _sigma[i]*_sigma[j] * _rho[i][j] * p(i)*p(j);
    }
  }

  return a_matrix;
}

template <int dim>
Tensor<1,dim>
AMatrix<dim>::divergence(const Point<dim> &p) const
{
  Tensor<1,dim> div_vector;

    for (unsigned int j=0; j<p.dimension; ++j)
    {
      for (unsigned int i=0; i<p.dimension; ++i)
      {
        div_vector[j] += 0.5*_sigma[i]*_sigma[j]*_rho[i][j] * (p[j] + p[i]*(i==j?1:0));  
      }
    }

    return div_vector;
}

// ## End of A Matrix ##







// ## Q Vector ##
template <int dim>
class QVector: public TensorFunction<1,dim>
{
public:
  QVector();

  virtual Tensor<1,dim>
  value(const Point<dim> &p) const override;
};

template <int dim>
QVector<dim>::QVector()
  : TensorFunction<1,dim>()
{}

template <int dim>
Tensor<1,dim>
QVector<dim>::value(const Point<dim> &p) const
{
  Tensor<1,dim> q_vector;

  for (unsigned int i = 0; i < dim; i++)
  {
    q_vector[i] = interest_rate * p[i];
  }

  return q_vector;
}

// ## End of QVector ##



// #### Initial and Boundary Conditions ####

// ## Initial Conditions ##
template <int dim>
class InitialConditions: public Function<dim>
{
public:
  InitialConditions();

  virtual double value(const Point<dim> &p,
                      const unsigned int component = 0) const override;
};

template <int dim>
InitialConditions<dim>::InitialConditions() {}

template <int dim>
double InitialConditions<dim>::value(const Point<dim> &p,
                                    const unsigned int component) const
{
  double retVal = 0;
#ifdef MMS
  (void ) component;
  for (int i = 0; i < dim; i++)
  {
    retVal += -Utilities::fixed_power<2,double>(p(i));
  }
  retVal += 6;
#else
  (void) component;
  for (int i = 0; i < dim; i++)
  {
    retVal += p(i);
  }
  
  retVal = std::max(retVal - strike_price, 0.0);
#endif

#ifdef RAND
  retVal += dis(gen);
#endif
  return retVal;
}

// ## End of Initial Conditions ##

/*******************************************************************************
********************* Container for Computed Solutions *************************
*******************************************************************************/
template <int dim>
struct ComputedSolution
{
  DoFHandler<dim> dof_handler;

  Vector<double> solutionVec;
};

/*******************************************************************************
************************ Left Boundary Condition 1D ****************************
*******************************************************************************/
class LeftBoundary1D: public Function<1>
{
public:
  LeftBoundary1D();

  virtual double value(const Point<1> &p,
                      const unsigned int component = 0) const override;
};

/********************** Left Boundary 1D Constructor **************************/
/******************************************************************************/
LeftBoundary1D::LeftBoundary1D()
{}

/************************ Left Boundary 1D 'value' ****************************/
/******************************************************************************/
double LeftBoundary1D::value(const Point<1> &p,
                            const unsigned int component) const
{
  (void) p;
  (void) component;
  double ret = 0.0;
#ifdef MMS
  ret += -Utilities::fixed_power<2, double>(this->get_time()) + 6;
#else
  ret = 0.0;
#endif

  return ret;
}

/*******************************************************************************
************************* Right Boundary Condition *****************************
*******************************************************************************/
template <int dim>
class RightBoundary: public Function<dim>
{
public:
  RightBoundary();

  virtual double value(const Point<dim> &p,
                      const unsigned int component = 0) const override;
};

/********************** Right Boundary Constructor ****************************/
/******************************************************************************/
template <int dim>
RightBoundary<dim>::RightBoundary()
{}

/********************** Right Boundary Constructor ****************************/
/******************************************************************************/
template <int dim>
double RightBoundary<dim>::value(const Point<dim> &p,
                                const unsigned int component) const
{
  (void) component;
  double ret =  0;
#ifdef MMS
  for (unsigned int d=0; d<dim; d++)
  {
    ret += -Utilities::fixed_power<2,double>(p(d)); 
  }
  ret += -Utilities::fixed_power<2,double>(this->get_time()) + 6;
#else
  for (unsigned int d=0; d<dim; ++d)
  {
    ret += p(d);
  }
  ret -= strike_price * exp((-interest_rate) * (this->get_time()));
#endif

  return ret;
}

/*******************************************************************************
************************ Right Hand Side Function ******************************
*******************************************************************************/
template <int dim>
class RightHandSide: public Function<dim>
{
public:
  RightHandSide(const Tensor<1,dim> sigma_vector);

  virtual double value(const Point<dim> &p,
                      const unsigned int component = 0) const override;

private:
  Tensor<1,dim> _sigma;
};

/***************************** RHS Constructor ********************************/
/******************************************************************************/
template <int dim>
RightHandSide<dim>::RightHandSide(const Tensor<1,dim> sigma_vector)
: _sigma(sigma_vector)
{}

/******************************* RHS 'value' **********************************/
/******************************************************************************/
template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                const unsigned int component) const
{
  double ret = 0.0;
#ifdef MMS
  (void) component;
  for (unsigned int d=0; d<dim; d++)
  {
    ret += -Utilities::fixed_power<2,double>(p(d) * _sigma[d]);
    ret += -Utilities::fixed_power<2,double>(p(d)) * interest_rate;
  }
  ret += this->get_time() * (2 + this->get_time() * interest_rate);
  ret -= 6 * interest_rate;
#else
  (void) p;
  (void) component;
  ret = 0.0;
#endif

  return ret;
}



/*******************************************************************************
************************* Black-Scholes Solver *********************************
*******************************************************************************/

/*********************** Template Solver Definition ***************************/
/******************************************************************************/
template <int dim>
class BlackScholesSolver
{
public:
  BlackScholesSolver(const uint32_t b_id,
                    const int finalDim = dim);

  void refine_grid();

  void create_mesh();
  void initialize_matrices();
  void build_matrices();

  void build_b_matrix(FEValues<dim> &fe_values,
                  std::vector<types::global_dof_index> &local_dof_indices,
                  FullMatrix<double> cell_matrix);
  void build_c_matrix(FEValues<dim> &fe_values,
                  std::vector<types::global_dof_index> &local_dof_indices,
                  FullMatrix<double> cell_matrix);
  
  void apply_boundary_ids();

  void create_lower_dim_solver(const uint32_t b_id,
            std::map<uint32_t, std::unique_ptr<BlackScholesSolver<dim-1>>> &ld_solvers);

  void setup_system();
  Vector<double> create_rhs_linear_system() const;
  Vector<double> create_forcing_terms(const double curr_time) const;
  SparseMatrix<double> create_system_matrix() const;
  void impose_boundary_conditions(const double curr_time);
  void create_problem(const double curr_time);

  // void setup_system();
  // void create_problem();
  void do_one_timestep();
  
  void process_solution();
  void write_convergence_table();

  void output_results(const double curr_time, const uint64_t time_step_number);

  void run(const bool isLast = false);
  
  Vector<double> solution;
private:
  Triangulation<dim> triangulation;

  Point<dim-1> project_point(const Point<dim> &p, const uint32_t b_id);
  
public:
  DoFHandler<dim> dof_handler;
private:
  FE_Q<dim>          fe;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;

  std::map<uint32_t, std::unique_ptr<BlackScholesSolver<dim-1>>> lower_dim_solvers;

  double current_time;

  const Tensor<2,dim> _rho;
  const Tensor<1,dim> _sigma;

  AMatrix<dim> a_matrix;
  QVector<dim> q_vector;

  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> b_matrix;
  SparseMatrix<double> c_matrix;
  
  SparseMatrix<double> system_matrix;

  Vector<double> system_rhs;

  uint64_t final_dim;

  ConvergenceTable convergence_table;
};

/************************* 0-D Solver Definition ******************************/
/******************************************************************************/
template<>
class BlackScholesSolver<0>
{
public:
  BlackScholesSolver(const uint32_t b_id,
                    const int finalDim = 0);

  void setup_system();
  void create_problem(const double curr_time);
  void do_one_timestep();
  void refine_grid();

};

/********************** Template Solver Constructor ***************************/
/******************************************************************************/
template <int dim>
BlackScholesSolver<dim>::BlackScholesSolver(const uint32_t b_id,
                                            const int finalDim)
  : dof_handler(triangulation)
  , fe(1)
  , current_time(0)
  , _rho(create_rho_matrix<dim,tensor_dim>(b_id, rho))
  , _sigma(create_sigma_vector<dim,tensor_dim>(b_id, sigma))
  , a_matrix(_sigma, _rho)
  , final_dim(finalDim)
{
  GridGenerator::hyper_cube(triangulation, 0.0, maximum_stock_price, true);
  triangulation.refine_global(initial_global_refinement);
}


/************************* 0-D Solver Constructor *****************************/
/******************************************************************************/
BlackScholesSolver<0>::BlackScholesSolver(const uint32_t b_id, const int finalDim)
{
  (void) finalDim;
  (void) b_id;
}

/**************************** 0-D 'refine_grid' *******************************/
/******************************************************************************/
void BlackScholesSolver<0>::refine_grid()
{}

/************************* Template 'refine_grid' *****************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::refine_grid()
{
  // Refine this grid
  triangulation.refine_global(1);

  // Refine the lower dimensional solutions
  for (const auto &lowerDimSolverTT : lower_dim_solvers)
  {
    // std::cout << "REFINING: " << lowerDimSolverTT.first << std::endl;
    lowerDimSolverTT.second->refine_grid();
  }
}

/************************* Template 'create_mesh' *****************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::create_mesh()
{
  dof_handler.distribute_dofs(fe);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ true);
  sparsity_pattern.copy_from(dsp);
}

/******************** Template 'initialize_matrices' **************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::initialize_matrices()
{
  b_matrix.reinit(sparsity_pattern);
  c_matrix.reinit(sparsity_pattern);
  mass_matrix.reinit(sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);
}

/************************ Template 'build_matrices' ***************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::build_matrices()
{
  // Some setup first
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  QGauss<dim>        quadrature_formula(fe.degree + 1);
  FEValues<dim>      fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Next, to build the mass matrix
  MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);

  // Build the 'B' matrix
  build_b_matrix(fe_values, local_dof_indices, cell_matrix);

  // Build the 'C' matrix
  build_c_matrix(fe_values, local_dof_indices, cell_matrix);

}

/*********************** Template 'build_b_matrix' ****************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::build_b_matrix(FEValues<dim> &fe_values,
                  std::vector<types::global_dof_index> &local_dof_indices,
                  FullMatrix<double> cell_matrix)
{
  for (const auto &cell: dof_handler.active_cell_iterators())
  {
    cell_matrix = 0.;
    fe_values.reinit(cell);
    for (const unsigned int q_index: fe_values.quadrature_point_indices())
    {
      for (const unsigned i : fe_values.dof_indices())
      {
        for (const unsigned j : fe_values.dof_indices())
        {
          cell_matrix(i,j) +=
            (
              fe_values.shape_grad(i, q_index) * // grad phi_i
              a_matrix.value(fe_values.quadrature_point(q_index)) * // A(S)
              fe_values.shape_grad(j, q_index) * // grad phi_j
              fe_values.JxW(q_index) // dx
            );
        }
      }
    }
    cell->get_dof_indices(local_dof_indices);
    for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
          b_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
      }
  }
}

/*********************** Template 'build_c_matrix' ****************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::build_c_matrix(FEValues<dim> &fe_values,
                  std::vector<types::global_dof_index> &local_dof_indices,
                  FullMatrix<double> cell_matrix)
{
  for (const auto &cell: dof_handler.active_cell_iterators())
  {
    cell_matrix = 0.;
    fe_values.reinit(cell);
    for (const unsigned int q_index: fe_values.quadrature_point_indices())
    {
      for (const unsigned i : fe_values.dof_indices())
      {
        for (const unsigned j : fe_values.dof_indices())
        {
          cell_matrix(i,j) +=
            (
              fe_values.shape_value(i, q_index) * // phi_i
              (
                a_matrix.divergence(fe_values.quadrature_point(q_index)) // divergence(A)
                - q_vector.value(fe_values.quadrature_point(q_index)) // q_vector(S)
              ) * 
              fe_values.shape_grad(j, q_index) * // grad phi_j
              fe_values.JxW(q_index) // dx
            );
        }
      }
    }
    cell->get_dof_indices(local_dof_indices);
    for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
          c_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
      }
  }
}

/********************* Template 'apply_boundary_ids' **************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::apply_boundary_ids()
{
  // std::cout << "DIM: " << dim << std::endl;
  for (const auto &cell : triangulation.active_cell_iterators())
  {
    for (auto &face : cell->face_iterators())
    {
      // If the face is on the boundary, then I need to check if it lies on the
      // "origin corner".
      if (face->at_boundary())
      {
        // First, I am going to set the boundary id to an 'invalid' one.
        // This is because, in case the face is at the boundary but not on the
        // corner, I can know by looking at the boundary id
        // If we're in the 1D case, then, set the invalid state to 1.
        face->set_boundary_id(0);

        // To check where the face is, I will look at it's center
        const auto center = face->center();
        for (unsigned int i=0; i<dim; i++)
        {
          // A value of zero indicates that the face is on the corner
          if ((std::fabs(center(i) - (0.0)) < 1e-12))
          {
            face->set_boundary_id(boundary_id(i, dim));
            
            // Only one element of 'center' can be zero.
            break;
          }
        }
      }
    }
  }
}

/******************* Template 'create_lower_dim_solver' ***********************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::create_lower_dim_solver(const uint32_t b_id,
                                std::map<uint32_t, std::unique_ptr<BlackScholesSolver<dim-1>>> &ld_solvers)
{
  // If the boundary id doesn't exist, then create it. Otherwise, just pass
  if (ld_solvers.find(b_id) == ld_solvers.end())
  {
    ld_solvers[b_id] = std::make_unique<BlackScholesSolver<dim-1>>(b_id, final_dim);
  }
  else
  {
    return;
  }
}

/*************************** 0-D 'setup_system' *******************************/
/******************************************************************************/
void BlackScholesSolver<0>::setup_system()
{}

/************************* Template 'setup_system' ****************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::setup_system()
{
  create_mesh();

  initialize_matrices();
  build_matrices();

  // Index through faces and determine which faces correspond to the
  // initial conditions
  apply_boundary_ids();

  // Initialize system and solution vectors
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  // Apply initial conditions
  VectorTools::interpolate(dof_handler,
                            InitialConditions<dim>(),
                            solution);
  
  // For each of the lower dimensional solutions, set up the system
  for (uint64_t boundary = 0; boundary < choose(final_dim, dim-1); boundary++)
  {
    uint32_t b_id = boundary_id(boundary, final_dim);
    create_lower_dim_solver(b_id, lower_dim_solvers);
    lower_dim_solvers[b_id]->setup_system();
  }
}

/******************** Template 'create_rhs_linear_system' *********************/
/******************************************************************************/
template <int dim>
Vector<double> BlackScholesSolver<dim>::create_rhs_linear_system() const
{
  // I will compute the right hand side of the linear system
  // This is:
  // \left(\left(1-k_nr\right)\mathbf{M} - \frac{1}{2}k_n\mathbf{B} - k_n\mathbf{C}\right)V^{n-1}
  Vector<double> ret_rhs;
  ret_rhs.reinit(dof_handler.n_dofs());


  // Set up vector to hold temporary multiplication result
  Vector<double> vmult_result;
  vmult_result.reinit(dof_handler.n_dofs());

  // Now, to compute the right hand side of the linear system
  mass_matrix.vmult(vmult_result, solution);
  ret_rhs.add(
    1 - 0.5 * (time_step * interest_rate), vmult_result
  );

  b_matrix.vmult(vmult_result, solution);
  ret_rhs.add(
    (-1) * 0.5 * time_step, vmult_result
  );

  c_matrix.vmult(vmult_result, solution);
  ret_rhs.add(
    (-1) * time_step, vmult_result
  );

  // Add the forcing terms

  return ret_rhs;
}

/********************** Template 'create_forcing_terms' ***********************/
/******************************************************************************/
template <int dim>
Vector<double> BlackScholesSolver<dim>::create_forcing_terms(const double curr_time) const
{
  // I will compute the forcing terms 
  // This is:
  // -k_nF^n
  Vector<double> ret_forcing_terms;
  Vector<double> tmp;
  tmp.reinit(dof_handler.n_dofs());
  ret_forcing_terms.reinit(dof_handler.n_dofs());

  // Compute the forcing terms
  RightHandSide<dim> rhs_function(_sigma);
  rhs_function.set_time(curr_time);

  VectorTools::create_right_hand_side(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      rhs_function,
                                      tmp);
  tmp *= (-1) * 0.5 * time_step;
  ret_forcing_terms += tmp;

  rhs_function.set_time(curr_time - time_step);

  VectorTools::create_right_hand_side(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      rhs_function,
                                      tmp);
  tmp *= (-1) * 0.5 * time_step;
  ret_forcing_terms += tmp;

  return ret_forcing_terms;

}

/********************** Template 'create_system_matrix' ***********************/
/******************************************************************************/
template <int dim>
SparseMatrix<double> BlackScholesSolver<dim>::create_system_matrix() const
{
  // I will compute the left hand side of the linear system
  // This is:
  // \mathbf{M} + \frac{1}{2}k_n\mathbf{B}
  SparseMatrix<double> ret_system_matrix;
  ret_system_matrix.reinit(sparsity_pattern);

  // Compute the left hand side of the linear system
  ret_system_matrix.add(
    1 + 0.5*time_step * interest_rate, mass_matrix
  );

  ret_system_matrix.add(
    0.5 * time_step, b_matrix
  );

  return ret_system_matrix;
}

/**************** Template 'interpolate_boundary_conditions' ******************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::impose_boundary_conditions(const double curr_time)
{
  (void) curr_time;
  std::map<types::global_dof_index, double> boundary_values;
  RightBoundary<dim> right_boundary_function;
  right_boundary_function.set_time(curr_time);

  for (const auto &lowerDimSolverTT: lower_dim_solvers)
  {
    // // Get a function for the lower dimensional solution
    // Functions::FEFieldFunction<dim-1> SolutionLD(lowerDimSolver.dof_handler,
    //                                             boundary.second);
    uint32_t b_id = lowerDimSolverTT.first;
    
    // Get a function for the lower dimensional solution
    Functions::FEFieldFunction<dim-1> SolutionLD(lowerDimSolverTT.second->dof_handler,
                                                  lowerDimSolverTT.second->solution);
    
    // Next, I need to apply this function to the corresponding boundary
    VectorTools::interpolate_boundary_values(dof_handler,
                                                /*Boundary Id=*/b_id,
                                                ScalarFunctionFromFunctionObject<dim>
                                                ([b_id, SolutionLD, this](const Point<dim> &p)
                                                  {
                                                    const Point<dim-1> p_along_axis = project_point(p, b_id);
                                                    const double boundary_value = SolutionLD.value(p_along_axis);
                                                    
                                                    return boundary_value;
                                                  }
                                                ),
                                                boundary_values);
  }
  // abort();
  VectorTools::interpolate_boundary_values(dof_handler,
                                                /*Boundary Id=*/0,
                                                right_boundary_function,
                                                boundary_values);

  MatrixTools::apply_boundary_values(boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}

/****************** 1-D 'impose_boundary_conditions' *********************/
/******************************************************************************/
template <>
void BlackScholesSolver<1>::impose_boundary_conditions(const double curr_time)
{
  RightBoundary<1> right_boundary_function;
  LeftBoundary1D left_boundary_function;
  right_boundary_function.set_time(curr_time);
  left_boundary_function.set_time(curr_time);
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                            1,
                                            left_boundary_function,
                                            boundary_values);
  VectorTools::interpolate_boundary_values(dof_handler,
                                            0,
                                            right_boundary_function,
                                            boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}

/*************************** 0-D 'create_problem' *****************************/
/******************************************************************************/
void BlackScholesSolver<0>::create_problem(const double curr_time)
{
  (void) curr_time;
}

/************************ Template 'create_problem' ***************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::create_problem(const double curr_time)
{
  // Create the problem for all of the lower dimensional solutions
  for (const auto &lowerDimSolverTT : lower_dim_solvers)
  {
    lowerDimSolverTT.second->create_problem(curr_time);
    // Then solve one timestep for each of them, so that they can be used
    // as boundary conditions
    lowerDimSolverTT.second->do_one_timestep();
  }

  // Set up vectors to hold temporary data
  Vector<double> forcing_terms;

  // Initialize these vectors
  forcing_terms.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  system_rhs = create_rhs_linear_system();
  system_rhs += create_forcing_terms(curr_time);

  // Next, create the system matrix that needs to be inverted at every timestep
  system_matrix = create_system_matrix();

  constraints.condense(system_matrix, system_rhs);

  impose_boundary_conditions(curr_time);
}

/*********************** Template 'do_one_timestep' ***************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::do_one_timestep()
{
  SolverControl                          solver_control(1000, 1e-12);
  SolverCG<Vector<double>>               cg(solver_control);
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.0);
  cg.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);
}

/************************** 0-D 'do_one_timestep' *****************************/
/******************************************************************************/
void BlackScholesSolver<0>::do_one_timestep()
{

}

/************************ Template 'process_solution' *************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::process_solution()
{
  #ifdef MMS
    Solution<dim> sol;
    sol.set_time(current_time);
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    sol,
                                    difference_per_cell,
                                    QGauss<dim>(fe.degree + 1),
                                    VectorTools::L2_norm);
    const double L2_error =
    VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    sol,
                                    difference_per_cell,
                                    QGauss<dim>(fe.degree + 1),
                                    VectorTools::H1_seminorm);
    const double H1_error =
    VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_seminorm);
    const QTrapezoid<1>     q_trapez;
    const QIterated<dim> q_iterated(q_trapez, fe.degree * 2 + 1);
    VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    sol,
                                    difference_per_cell,
                                    q_iterated,
                                    VectorTools::Linfty_norm);
    const double Linfty_error =
    VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::Linfty_norm);

    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs         = dof_handler.n_dofs();
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
    convergence_table.add_value("Linfty", Linfty_error);
  #else
    return;
  #endif
}

/********************* Template 'write_convergence_table' *********************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::write_convergence_table()
{
  #ifdef MMS
    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("H1", 3);
    convergence_table.set_precision("Linfty", 3);
    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);
    convergence_table.set_scientific("Linfty", true);
    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("L2", "@f@f$L^2@f@f$-error");
    convergence_table.set_tex_caption("H1", "@f@f$H^1@f@f$-error");
    convergence_table.set_tex_caption("Linfty", "@f@f$L^\\infty@f@f$-error");
    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");
    std::cout << std::endl;
    convergence_table.write_text(std::cout);
    std::string error_filename = "error";
    error_filename += "-global";
    error_filename += ".tex";
    std::ofstream error_table_file(error_filename);
    convergence_table.write_tex(error_table_file);

    convergence_table.add_column_to_supercolumn("cells", "n cells");
    std::vector<std::string> new_order;
    new_order.emplace_back("n cells");
    new_order.emplace_back("H1");
    new_order.emplace_back("L2");
    convergence_table.set_column_order(new_order);
    convergence_table.evaluate_convergence_rates(
      "L2", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "H1", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "H1", ConvergenceTable::reduction_rate_log2);
    std::cout << std::endl;
    convergence_table.write_text(std::cout);
    std::string conv_filename = "convergence";
    conv_filename += "-global";
    switch (fe.degree)
      {
        case 1:
          conv_filename += "-q1";
          break;
        case 2:
          conv_filename += "-q2";
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
    conv_filename += ".tex";
    std::ofstream table_file(conv_filename);
    convergence_table.write_tex(table_file);
  #else
    return;
  #endif
}

/*********************** Template 'output_results' ****************************/
/******************************************************************************/
template<int dim>
void BlackScholesSolver<dim>::output_results(const double curr_time,
                                            const uint64_t time_step_number)
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "U");
  data_out.build_patches();
  data_out.set_flags(DataOutBase::VtkFlags(curr_time, time_step_number));
  const std::string filename =
    "solution-" + std::to_string(time_step_number) + ".vtu";
  std::ofstream output(filename);
  data_out.write_vtu(output);
}


/***************************** Template 'run' *********************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::run(const bool isLast)
{
  current_time = 0.0;
  setup_system();

  // First time
  create_problem(current_time);
  do_one_timestep();
  
  if (isLast)
  {
    output_results(current_time, 0);
  }

  for (uint64_t time_step_number = 0; time_step_number < n_time_steps; time_step_number++)
  {
    current_time += time_step;

    if (time_step_number % 10 == 0)
    {
      double percentDone = ((double)time_step_number / (double)n_time_steps) * 100;
      std::cout << "TIME STEP NUMBER: " << time_step_number << ". PERCENT DONE: " << percentDone << "%" << std::endl;
    }

    create_problem(current_time);
    do_one_timestep();

    if (isLast)
    {
      output_results(current_time, time_step_number + 1);
    }

  }
}

/************************ Template 'project_point' ****************************/
/******************************************************************************/
template <int dim>
Point<dim-1> 
BlackScholesSolver<dim>::project_point(const Point<dim> &p, const uint32_t b_id)
{
  Point<dim-1> retPoint;
  uint64_t retPointIdx = 0;

  uint32_t b_id_copy = b_id;


  for (uint64_t pointIdx = 0; pointIdx < dim; pointIdx++)
  {
    if (b_id_copy & 1)
    {
      retPoint[retPointIdx] = p[pointIdx];
      retPointIdx++;
    }

    b_id_copy >>= 1;
  }

  return retPoint;
}



int main()
{
  uint32_t full_bits = (1 << dimension) - 1;

  BlackScholesSolver<dimension> bsp(full_bits, dimension);

  #ifdef MMS
  unsigned int num_cycles = 6;

  // bsp.run();
  // bsp.output_results(maturity_time);
  for (unsigned int cycle = 0; cycle < num_cycles; cycle++)
  {
    std::cout << "CYCLE: " << cycle << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    bsp.run(cycle == num_cycles-1);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    bsp.process_solution();

    bsp.refine_grid();
  }
  bsp.write_convergence_table();
  #endif

  #ifndef MMS
    bsp.run(true);
  #endif
}