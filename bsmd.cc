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
  const unsigned int dimension = 3;
  const unsigned int n_time_steps = 5;
  const double maturity_time = 1.0;
  double time_step = maturity_time / n_time_steps;
  unsigned int timestep_number = 0;
  double current_time = 0.0; 
  const double maximum_stock_price = 5.0;
  const double strike_price = 0.5;
  const Tensor<2, dimension> rho({ {1,.5,.6}
                                    ,{.5,1,.7}
                                    ,{.6,.7,1}});
  const Tensor<1, dimension> sigma( {.2,.2,.3} );
  const double interest_rate = .05;
  #define MMS
// ############### End setting parameters ###################

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



int main()
{
  Solution<dimension> sol;
  Point<dimension> p = {1,2,3};
  Tensor<1,dimension> test = sol.gradient(p);

  std::cout << test[1] << std::endl;
}