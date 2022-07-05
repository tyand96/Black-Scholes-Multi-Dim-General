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

using namespace dealii;



// ############### Begin setting parameters ###################
  constexpr unsigned int dimension = 1;
  const unsigned int n_time_steps = 100;
  const double maturity_time = 1.0;
  double time_step = maturity_time / n_time_steps;
  // unsigned int timestep_number = 0;
  // double current_time = 0.0; 
  const double maximum_stock_price = 5.0;
  const double strike_price = 0.5;
  // const Tensor<2, dimension> rho({ {1,.5,.6}
  //                                   ,{.5,1,.7}
  //                                   ,{.6,.7,1}});
  const Tensor<2,1> rho({{1}});
  // const Tensor<1, dimension> sigma( {.2,.2,.3} );
  const Tensor<1,1> sigma({.2});
  const double interest_rate = .05;
  constexpr unsigned int initial_global_refinement = 0;
  #define MMS
// ############### End setting parameters ###################

// ## Helper Methods ##
uint64_t choose(uint64_t n, uint64_t k)
{
  if (k == 0)
  {
    return 1;
  }
  return (n * choose(n-1, k-1) / k);
}

// ### MMS Solution ###
#ifdef MMS
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

// Constructor
template <int dim>
Solution<dim>::Solution()
{}

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
  AMatrix();

  virtual Tensor<2,dim>
  value(const Point<dim> &p) const override;

  Tensor<1,dim>
  divergence(const Point<dim> &p) const;
};

// Constructor
template <int dim>
AMatrix<dim>::AMatrix()
  : TensorFunction<2,dim>()
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
      a_matrix[i][j] = 0.5 * sigma[i]*sigma[j] * rho[i][j] * p(i)*p(j);
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
        div_vector[j] += 0.5*sigma[i]*sigma[j]*rho[i][j] * (p[j] + p[i]*(i==j?1:0));  
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
  Assert(false, ExcNotImplemented());
  retVal = 0.0;
#endif
  return retVal;
}

// ## End of Initial Conditions ##

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
  ret = -Utilities::fixed_power<2, double>(this->get_time()) + 6;
#else
  Assert(false, ExcNotImplemented());
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
  Assert(false, ExcNotImplemented());
  // for (unsigned int d=0; d<dim; ++d)
  // {
  //   ret += p(d);
  // }
  // ret -= strike_price * exp((-interest_rate) * (this->get_time()));
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
  RightHandSide();

  virtual double value(const Point<dim> &p,
                      const unsigned int component = 0) const;

};

/***************************** RHS Constructor ********************************/
/******************************************************************************/
template <int dim>
RightHandSide<dim>::RightHandSide()
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
    ret += -Utilities::fixed_power<2,double>(p(d) * sigma[d]);
    ret += -Utilities::fixed_power<2,double>(p(d)) * interest_rate;
  }
  ret += this->get_time() * (2 + this->get_time() * interest_rate);
  ret += 6 * interest_rate;
#else
  Assert(false, ExcNotImplemented());
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
  BlackScholesSolver();

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

  void setup_system();
  Vector<double> create_rhs_linear_system() const;
  Vector<double> create_forcing_terms(const double curr_time) const;
  SparseMatrix<double> create_system_matrix() const;
  void impose_boundary_conditions();
  void create_problem(const double curr_time);

  // void setup_system();
  // void create_problem();
  std::map<uint64_t, Vector<double>> do_one_timestep(const double current_time,
                                                  const int finalProblemDim);
  
  void process_solution(const double curr_time);
  void write_convergence_table();

  void run();
  
  Vector<double> solution;
private:
  Triangulation<dim> triangulation;
public:
  DoFHandler<dim> dof_handler;
private:
  FE_Q<dim>          fe;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;

  BlackScholesSolver<dim-1> lowerDimSolver;

  double current_time;

  AMatrix<dim> a_matrix;
  QVector<dim> q_vector;

  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> b_matrix;
  SparseMatrix<double> c_matrix;
  
  SparseMatrix<double> system_matrix;

  Vector<double> system_rhs;

  ConvergenceTable convergence_table;
};

/************************* 0-D Solver Definition ******************************/
/******************************************************************************/
template<>
class BlackScholesSolver<0>
{
public:
  BlackScholesSolver();

};

/********************** Template Solver Constructor ***************************/
/******************************************************************************/
template <int dim>
BlackScholesSolver<dim>::BlackScholesSolver()
  : dof_handler(triangulation)
  , fe(1)
  , current_time(0)
{
  GridGenerator::hyper_cube(triangulation, 0.0, maximum_stock_price, true);
  triangulation.refine_global(initial_global_refinement);
}


/************************* 0-D Solver Constructor *****************************/
/******************************************************************************/
BlackScholesSolver<0>::BlackScholesSolver()
{}

/************************* Template 'refine_grid' *****************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::refine_grid()
{
  triangulation.refine_global(1);
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
        face->set_boundary_id(0);

        // To check where the face is, I will look at it's center
        const auto center = face->center();
        for (unsigned int i=0; i<dim; i++)
        {
          // A value of zero indicates that the face is on the corner
          if ((std::fabs(center(i) - (0.0)) < 1e-12))
          {
            face->set_boundary_id(1<<i);
            
            // Only one element of 'center' can be zero.
            break;
          }
        }
      }
      // std::cout << (int64_t)face->boundary_id() << ": " << face->at_boundary() << std::endl;
    }
  }
}

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
    1 - (time_step * interest_rate), vmult_result
  );

  b_matrix.vmult(vmult_result, solution);
  ret_rhs.add(
    (-1) * time_step, vmult_result
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
  ret_forcing_terms.reinit(dof_handler.n_dofs());

  // Compute the forcing terms
  RightHandSide<dim> rhs_function;
  rhs_function.set_time(curr_time);

  VectorTools::create_right_hand_side(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      rhs_function,
                                      ret_forcing_terms);
  ret_forcing_terms *= (-1) * time_step;

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
    1, mass_matrix
  );

  ret_system_matrix.add(
    0.5 * time_step, b_matrix
  );

  return ret_system_matrix;
}

/**************** Template 'interpolate_boundary_conditions' ******************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::impose_boundary_conditions()
{
  Assert(false, ExcNotImplemented());
}

/****************** 1-D 'impose_boundary_conditions' *********************/
/******************************************************************************/
template <>
void BlackScholesSolver<1>::impose_boundary_conditions()
{
  RightBoundary<1> right_boundary_function;
  LeftBoundary1D left_boundary_function;
  right_boundary_function.set_time(current_time);
  left_boundary_function.set_time(current_time);
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

/************************ Template 'create_problem' ***************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::create_problem(const double curr_time)
{
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

  impose_boundary_conditions();
}

/*********************** Template 'do_one_timestep' ***************************/
/******************************************************************************/
template <int dim>
std::map<uint64_t, Vector<double>>
BlackScholesSolver<dim>::do_one_timestep(const double curr_time,
                                        const int finalProblemDim)
{
  std::map<uint64_t, Vector<double>> lowerDimSol = lowerDimSolver.do_one_timestep(curr_time, dim);
  Vector<double> vec({4,5,6});
  std::map<uint64_t, Vector<double>> ret;

  uint64_t num_solutions = choose(finalProblemDim, dim);

  for (uint64_t i = 0; i < num_solutions; i++)
  {
    SolverControl                          solver_control(1000, 1e-12);
    SolverCG<Vector<double>>               cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
    cg.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

    ret[(1<<i)] = solution;
  }

  return ret;
}

/************************** 1-D 'do_one_timestep' *****************************/
/******************************************************************************/
template<>
std::map<uint64_t, Vector<double>>
BlackScholesSolver<1>::do_one_timestep(const double curr_time,
                                      const int finalProblemDim)
{
  (void) curr_time;
  std::map<uint64_t, Vector<double>> ret;

  uint64_t num_solutions = choose(finalProblemDim, 1);

  for (uint64_t i = 0; i < num_solutions; i++)
  {
    SolverControl                          solver_control(1000, 1e-12);
    SolverCG<Vector<double>>               cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
    cg.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

    ret[(1<<i)] = solution;
  }

  // std::cout << "1D currTime: " << curr_time << "... ";
  // for (const auto &p: ret[1])
  // {
  //   std::cout << p << " ";
  // }
  // std::cout << std::endl;

  return ret;
}

/************************ Template 'process_solution' *************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::process_solution(const double curr_time)
{
  Solution<dim> sol;
  sol.set_time(curr_time);
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
  const QTrapez<1>  q_trapezoid;
  const QIterated<dim> q_iterated(q_trapezoid, fe.degree * 2 + 1);
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
}

/********************* Template 'write_convergence_table' *********************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::write_convergence_table()
{
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
}


/***************************** Template 'run' *********************************/
/******************************************************************************/
template <int dim>
void BlackScholesSolver<dim>::run()
{
  setup_system();

  while (current_time < maturity_time)
  {
    create_problem(current_time);
    do_one_timestep(current_time, dim);

    current_time += time_step;
  }
}



int main()
{
  Solution<dimension> sol;
  // Point<dimension> p = {1,2,3};
  Point<dimension> p(5);
  std::cout << p(0) << std::endl;
  Tensor<1,dimension> test = sol.gradient(p);

  std::cout << test[0] << std::endl;


  AMatrix<dimension> amatrixFunc;
  
  Tensor<2,dimension> amatrix = amatrixFunc.value(p);
  std::cout << amatrix[0][0] << std::endl;

  BlackScholesSolver<dimension> bsp;
  for (unsigned int cycle = 0; cycle < 8; cycle++)
  {
    bsp.refine_grid();

    bsp.run();

    bsp.process_solution(maturity_time);
  }
  bsp.write_convergence_table();
  // auto r = bsp.do_one_timestep(0,dimension);
  // std::cout << r[1][0] << std::endl;
  // bsp.apply_boundary_ids();

  // // BlackScholesSolver<2> testSolver;
  // // testSolver.apply_boundary_ids();
  // bsp.run();
}