/* -----------------------------------------------------------------------
 * This program builds a block matrix that corresponds to one Newton
 * step in an inverse scattering problem (in 2D) along with the
 * corresponding right hand side. The code from step-20, step-22, step-21,
 * and step-15 of the deal.ii software package is used as a foundation
 * for this program.
 * -----------------------------------------------------------------------

 *
 * Author: Scott Ziegler, Colorado State University, 2020
 */


// @sect3{Include files}

// These include files declare the classes
// which handle triangulations and enumeration of degrees of freedom:

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>

// And this is the file in which the functions are declared that create grids:

#include <deal.II/grid/grid_generator.h>

// Output of grids in various graphics formats:

#include <deal.II/grid/grid_out.h>

// The next file contains classes which are needed for loops over all
// cells and to get the information from the cell objects. tria_iterator is
// used to get geometric information from cells:

#include <deal.II/grid/tria_iterator.h>

// These files contains the description of the finite elements
// used in this program:

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>

// And these files are needed for the creation of sparsity patterns of sparse
// matrices and renumbering of degrees of freedom:

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

// The next three files are needed for assembling the block matrix using quadrature on
// each cell:

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

// The following three include files we need for the treatment of boundary
// values:

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

// This group of files is for the linear algebra which we employ to
// solve the system of equations arising from the finite element discretization
// of the Helmholtz equation. The first two files are for dealing with
// the block structure that arises from the discretization in our problem. The next
// four files are for the local contributions to the block matrix.

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>

// This include file is needed for C++ output:

#include <fstream>

// This include file allows the program to be timed:

#include <ctime>

// Finally, this is for output to a file and to the console:

#include <iostream>

/**
 * The primary namespace for the inverse acoustic
 * scattering solver. InverseAcoustic contains all other namespaces,
 * classes, and functions needed to solve the inverse acoustic
 * scattering problem. The main components of InverseAcoustic are the HelmholtzSolver
 * class and the PrescribedSolution namespace. This class and namespace
 * contain all information necessary for constructing the linear system
 * corresponding to a single Newton step and solving that system to produce
 * an update direction.
 */

namespace InverseAcoustic {

// We include the namespace dealii:

    using namespace dealii;

// The main class for this problem is HelmholtzSolver. This
// class has two public functions, a constructor and a run function.
// Later, in the main() function, we will simply need to call the
// constructor and run function. The private member functions
// make_grid_and_dofs() and assemble_system() will be called in the
// run() function and are explained below as they are defined.

/**
 * This class defines all variables and functions needed to assemble
 * and solve the inverse acoustic scattering problem.
 */

    template <int dim>
    class HelmholtzSolver {
    public:
        HelmholtzSolver(const unsigned int degree, const unsigned int n_frequencies, const unsigned int n_transducers, const int regularization_parameter);
        void run();

    private:
        void make_grid_and_dofs();
        void assemble_system();

        // The following constants set the dimension of the space in
        // which the domain lives, the base degree
        // of the finite element spaces, the number of frequencies
        // transmitted, the number of transducers used to transmit
        // and measure signals, the number of experiments (which is
        // the number of frequencies multiplied by the number of
        // transducers), and the regularization parameter beta:

        const unsigned int degree; //!< Degree of freedom for the state and adjoint variables.
        const unsigned int n_frequencies; //!< Number of frequencies transmitted.
        const unsigned int n_transducers; //!< Number of transducers used to transmit and measure signals.
        const unsigned int n_experiments; //!< Total number of experiments (n_frequencies*n_transducers).
        const int regularization_parameter; //!< The regularization parameter.

        // This variable stores the number of refinement steps for
        // our grid:

        const unsigned int n_refinement_steps; //!< Number of times the grid is refined.

        // The following three objects are used in the
        // construction of our mesh and distribution of DOFs
        // on that mesh. The <2> template entry specifies that
        // we are working in 2D space:

        Triangulation<dim> triangulation; //!< Describes how we will refine the domain.
        FESystem<dim> fe; //!< Describes the kind of shape functions we want to use.
        DoFHandler<dim> dof_handler;

        // The next four member variables will be used
        // to set the block matrix that forms the linear system
        // for our problem:

        BlockSparsityPattern sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        BlockVector<double> solution;
        BlockVector<double> system_rhs;

        // Finally, the last two member variables store the previous
        // step of the Newton method as well as the current step:

        BlockVector<double> old_solution; //!< An object of type BlockVector which stores the previous Newton step.
        Vector<double> present_solution;
    };

    /**
     * This namespace assigns the right hand side of the Helmholtz equation and
     * the initial pressure values on the boundary.
     */

    namespace PrescribedSolution
    {

        /**
         * This class sets assigns the right hand side of the Helmholtz equation. RightHandSide
         * inherits from the Function<2> class from the deal.ii library. It contains
         * a constructor and a public member function RightHandSide::value which takes in a point in
         * 2D space and the (component?) for that point. The resulting output is the
         * right hand side of the Helmholtz equation representing any acoustic sinks or sources
         * in the domain.
         */

        template <int dim>
        class RightHandSide : public Function<2>
        {
        public:
            RightHandSide()
                    : Function<dim>(1) {}

            virtual double value(const Point<dim> &p,
                                 const unsigned int component = 0) const override;
        };

        /**
         * This class sets assigns the pressure values on the boundary of the domain. PressureBoundaryValues
         * inherits from the Function<2> class from the deal.ii library.
         */

        template <int dim>
        class PressureBoundaryValues : public Function<dim>
        {
        public:
            PressureBoundaryValues()
                    : Function<dim>(1) {}

            virtual double value(const Point<dim> &p,
                                 const unsigned int component = 0) const override;
        };

        // Set the right hand side of the Helmholtz equation to 0, meaning
        // there are no acoustic sources or sinks in the domain:

        template <int dim>
        double RightHandSide<dim>::value(const Point<dim> & /*p*/,
                                         const unsigned int /*component*/) const
        {
            return 0;
        };

        // Temporarily set the boundary values on the pressure to 1:

        template <int dim>
        double PressureBoundaryValues<dim>::value(const Point<dim> &p,
                                           const unsigned int /*component*/) const
        {
            return 1;
        };
    }

    /**
     * The constructor for the HelmholtzSolver class, this function takes four parameters and instantiates
     * their corresponding private member variables. After obtaining n_frequencies and n_transducers, it sets
     * the member variable n_experiments to be the product of these two quantities. In addition, the constructor
     * assigns the finite element, the number of refinements performed on the grid, and instantiates the dof handler
     * with the triangulation over the mesh. When assigning fe, we must assign a finite element for each of the
     * state variables (of which there are n_experiment copies), each of the adjoint variables (again, n_experiment
     * copies), and the parameter. We assign all the state and adjoin variables to have a finite element of degree
     * "degree" and the parameter to have a piecewise constant finite element.
     * @param degree
     * @param n_frequencies
     * @param n_transducers
     * @param regularization_parameter
     */

    template <int dim>
    HelmholtzSolver<dim>::HelmholtzSolver(const unsigned int degree, const unsigned int n_frequencies,
                                     const unsigned int n_transducers, const int regularization_parameter)
            : degree(degree)
            , n_frequencies(n_frequencies)
            , n_transducers(n_transducers)
            , n_experiments(n_frequencies * n_transducers)
            , regularization_parameter(regularization_parameter)
            , fe(FE_Q<dim>(degree), n_experiments, FE_Q<dim>(degree), n_experiments, FE_DGQ<dim>(0), 1)
            , n_refinement_steps(2)
            , dof_handler(triangulation)
            {}

    /**
     * This member function creates the domain, the mesh over that domain, and assigns degrees of freedom
     * to the corresponding mesh. The domain is set to be the unit disc and is refined using the refinement
     * steps set in the constructor for the HelmholtzSolver class. The degrees of freedom are then assigned
     * to this mesh and the sparsity pattern of the matrix corresponding to one update in a Newton algorithm
     * is set using these degrees of freedom.
     */

    template <int dim>
    void HelmholtzSolver<dim>::make_grid_and_dofs()
    {
        GridGenerator::hyper_ball(triangulation);
        triangulation.refine_global(n_refinement_steps);
        dof_handler.distribute_dofs(fe);

        // Check that I understand what is happening in these lines.
        DoFRenumbering::component_wise(dof_handler);
        const std::vector<types::global_dof_index> dofs_per_component =
                DoFTools::count_dofs_per_fe_component(dof_handler);

        // Now figure out how many degrees of freedom are associated with
        // each variable in our solution vector. The first n_experiments entries
        // of our solution vector are the state variables for each experiment, the
        // next n_experiments entries of our solution vector are the adjoint
        // variables, and the last entry of our solution vector is the scalar entry
        // representing our parameter of interest:

        const unsigned int d_p_per_experiment = dofs_per_component[0],
                d_l_per_experiment = dofs_per_component[n_experiments],
                d_g = dofs_per_component[2*n_experiments];

        // Print the number of active cells at each refinement state as well as
        // the total number of degrees of freedom and number of degrees of
        // freedom for the state variables, adjoint variables, and parameter gamma.:

        std::cout << "Number of active cells: " << triangulation.n_active_cells()
                  << std::endl
                  << "Number of degrees of freedom: " << dof_handler.n_dofs()
                  << " (" << d_p_per_experiment
                  << "x" << n_experiments
                  << " + " << d_l_per_experiment
                  << "x" << n_experiments
                  << " + " << d_g << ')' << std::endl
                  << std::endl;

        // We now assign the sparsity pattern for our block matrix based
        // on the degrees of freedom above for each component of our vector. Since
        // we set the finite elements to be the same for the state and adjoint
        // variables, we omit any d_l_per_experiment and replace it with
        // d_p_per_experiment:

        BlockDynamicSparsityPattern dsp(2*n_experiments+1, 2*n_experiments+1);
        for (unsigned int block_i=0; block_i<2*n_experiments; ++block_i)
        {
            for (unsigned int block_j=0; block_j<2*n_experiments; ++block_j)
                dsp.block(block_i, block_j).reinit(d_p_per_experiment, d_p_per_experiment);
        }

        for (unsigned int block_i=0; block_i<2*n_experiments; ++block_i)
            dsp.block(block_i,2*n_experiments).reinit(d_p_per_experiment, d_g);
        for (unsigned int block_j=0; block_j<2*n_experiments; ++block_j)
            dsp.block(2*n_experiments,block_j).reinit(d_g, d_p_per_experiment);

        dsp.block(2*n_experiments,2*n_experiments).reinit(d_g,d_g);

        // Use the compressed block sparsity pattern in the same way to create
        // the sparsity pattern for the full system matrix:

        dsp.collect_sizes();
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);

        // Print the sparsity pattern as a .gnu file:

        std::ofstream out("sparsity_pattern1.gnu");
        sparsity_pattern.print_gnuplot(out);

        // Finally resize the solution, old solution, and right hand side vectors in exactly the
        // same way as the block compressed sparsity pattern:

        solution.reinit(2*n_experiments+1);
        for (unsigned int block_i=0; block_i<2*n_experiments; ++block_i)
            solution.block(block_i).reinit(d_p_per_experiment);
        solution.block(2*n_experiments).reinit(d_g);
        solution.collect_sizes();

        old_solution.reinit(2*n_experiments+1);
        for (unsigned int block_i=0; block_i<2*n_experiments; ++block_i)
            old_solution.block(block_i).reinit(d_p_per_experiment);
        old_solution.block(2*n_experiments).reinit(d_g);
        old_solution.collect_sizes();

        system_rhs.reinit(2*n_experiments+1);
        for (unsigned int block_i=0; block_i<2*n_experiments; ++block_i)
            system_rhs.block(block_i).reinit(d_p_per_experiment);
        system_rhs.block(2*n_experiments).reinit(d_g);
        system_rhs.collect_sizes();
    }

    /**
     * This function assigns the local contributions to the system matrix on each cell of our grid. It begins
     * by specifying the quadrature used to calculate integrals and the values we need from each shape function.
     * It then cycles through each cell (at each experiment) and calculates the necessary terms from the bilinear
     * form on each cell. It assigns these terms to the local matrix and right hand side then keeps track of where
     * the degrees of freedom for that cell are located in the full system matrix.
     */

    template <int dim>
    void HelmholtzSolver<dim>::assemble_system()
    {
        QGauss<dim> quadrature_formula(degree + 1); // what degree should I choose here?
        QGauss<dim-1> face_quadrature_formula(degree + 1);

        // Specify the quantities we want to update for the shape functions in our
        // finite elements. In this case, we want to update
        FEValues<dim> fe_values(fe,
                              quadrature_formula,
                              update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
        FEFaceValues<dim> fe_face_values(fe,
                                       face_quadrature_formula,
                                       update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

        // Store the number of degrees of freedom on each cell, the number of quadrature
        // points in the interior and the number of quadrature points on the boundary:

        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        // For the local contributions we have an object of type FullMatrix and
        // a right hand side which will be of size dofs_per_cell on each cell:

        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // We will need to keep track of the local degree of freedom indices in
        // order to properly place the local contribution into the full sparse matrix:

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        // Extract the right hand side and initial pressure values from the PrescribedSolution
        // namespace:

        const PrescribedSolution::RightHandSide<dim> right_hand_side;
        const PrescribedSolution::PressureBoundaryValues<dim> pressure_boundary_values;

        // The local right hand side and boundary values will be vectors of
        // size n_q_points and n_face_q_points, respectively:

        std::vector<double> rhs_values(n_q_points);
        std::vector<double> boundary_values(n_face_q_points);

        // Fill the vectors state_variables and adjoint_variables with their
        // corresponding finite element indices. This is simply a way for us
        // to label the finite elements in a way that they align with our
        // understanding of the state, adjoint, and parameter values:

        std::vector<FEValuesExtractors::Scalar> state_variables(n_experiments);
        for (unsigned int i = 0; i < n_experiments; ++i)
            state_variables[i] = FEValuesExtractors::Scalar(i);
        std::vector<FEValuesExtractors::Scalar> adjoint_variables(n_experiments);
        for (unsigned int i = 0; i < n_experiments; ++i)
            adjoint_variables[i] = FEValuesExtractors::Scalar(n_experiments + i);
        FEValuesExtractors::Scalar parameter_gamma(2 * n_experiments);

        // We'll store the old solution values and old solution gradients in the following
        // objects then get the old function gradients by calling the get_function_gradients
        // method from the FEValuesBase class:

        // NOTE: FOR THE MOMENT WE ARE IGNORING THE PROPER DEFINITION OF M^i. ASSUME FOR THE
        // MOMENT THAT IT IS THE IDENTITY OPERATOR.

        std::vector<Vector<double>> old_solution_values(n_q_points, Vector<double>(2*n_experiments+1));
        std::vector<std::vector<Tensor<1, dim>>> old_solution_gradients(n_q_points, std::vector<Tensor<1,dim>>(2*n_experiments+1));

        // Loop over all active cells:

        for (const auto &cell : dof_handler.active_cell_iterators())
        {

            // Reinitialize the local matrices on each cell:

            fe_values.reinit(cell);
            local_matrix = 0;
            local_rhs = 0;
            right_hand_side.value_list(fe_values.get_quadrature_points(),
                                       rhs_values);

            // Obtain the function values and gradients from the previous Newton step:

            fe_values.get_function_values(old_solution,
                                             old_solution_values);
            fe_values.get_function_gradients(old_solution,
                                             old_solution_gradients);

            // Loop over the total number of experiments:

            for (unsigned int k = 0; k < n_experiments; ++k)
            {

                // Loop over all quadrature points:

                for (unsigned int q = 0; q < n_q_points; ++q)
                {

                    // Name all relevant variables which appear in the bi-linear form for the matrix and right-hand
                    // side:

                    const double old_state = old_solution_values[q][k];
                    const double old_adjoint = old_solution_values[q][n_experiments + k];
                    const double old_parameter = old_solution_values[q][2 * n_experiments];
                    const Tensor<1, dim> old_state_gradient = old_solution_gradients[q][k];
                    const Tensor<1, dim> old_adjoint_gradient = old_solution_gradients[q][n_experiments + k];
                    const Tensor<1, dim> old_parameter_gradient = old_solution_gradients[q][2*n_experiments];

                    // Loop over all shape function indices i and j:

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {

                        // Label relevant shape function values:

                        const double phi_i_s = fe_values[state_variables[k]].value(i,q);
                        const Tensor<1, dim> grad_phi_i_s = fe_values[state_variables[k]].gradient(i, q);
                        const double phi_i_a = fe_values[adjoint_variables[k]].value(i,q);
                        const Tensor<1, dim> grad_phi_i_a = fe_values[adjoint_variables[k]].gradient(i, q);
                        const double phi_i_g = fe_values[parameter_gamma].value(i,q);

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            const double phi_j_s = fe_values[state_variables[k]].value(j, q);
                            const Tensor<1, dim> grad_phi_j_s = fe_values[state_variables[k]].gradient(j, q);
                            const double phi_j_a = fe_values[adjoint_variables[k]].value(j, q);
                            const Tensor<1, dim> grad_phi_j_a = fe_values[adjoint_variables[k]].gradient(j, q);
                            const double phi_j_g = fe_values[parameter_gamma].value(j,q);

                            // Add the contributions from the bilinear form to our local matrix:

                            local_matrix(i, j) +=
                                    (phi_i_s * phi_j_s + // This term will need to change when I add the operator M^i.
                                    grad_phi_i_s * grad_phi_j_s +
                                    phi_i_s * old_parameter * phi_j_s +
                                    grad_phi_i_a * grad_phi_j_a +
                                    old_parameter * phi_i_a * phi_j_a +
                                    old_state * phi_i_a * phi_j_a +
                                    phi_i_g * old_state * phi_j_g)
                                    *fe_values.JxW(q);
                        }

                        // Add the contributions from the bilinear form to our local right hand side:

                        local_rhs(i) += (grad_phi_i_s * old_state_gradient +
                                phi_i_s * old_parameter * old_state -
                                phi_i_s * rhs_values[q] +
                                grad_phi_i_a * old_adjoint_gradient +
                                old_parameter * phi_i_a * old_adjoint +
                                phi_i_a * old_state + // This term will need to change when I add the M^i and z^i terms.
                                phi_i_g * old_state * old_adjoint)
                                * fe_values.JxW(q);
                    }
                }
            }

            // The following nested for loops provide the contribution of the regularization
            // term. This should not be summed over the number of experiments, so we remove it
            // from the "k" for loop over the number of experiments.

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                const double old_parameter = old_solution_values[q](2 * n_experiments);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {

                    const double phi_i_g = fe_values[parameter_gamma].value(i,q);

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {

                        const double phi_j_g = fe_values[parameter_gamma].value(j,q);

                        local_matrix(i, j) +=
                                (regularization_parameter * phi_i_g * phi_j_g)
                                *fe_values.JxW(q);
                        local_rhs(i) +=
                                regularization_parameter * phi_i_g * old_parameter;
                    }
                }
            }

            // Loop over active cell faces:

            for (const auto &face : cell->face_iterators())
                if (face->at_boundary()) {
                    fe_face_values.reinit(cell, face);
                    pressure_boundary_values.value_list(fe_face_values.get_quadrature_points(), boundary_values);
                    for (unsigned int k = 0; k < n_experiments; ++k)
                    {
                        for (unsigned int q = 0; q < n_face_q_points; ++q)
                        {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                local_rhs(i) += fe_face_values.JxW(q);
                        }
                    }
                }

            // Keep track of degree of freedom indices on each cell and add these
            // local contributions correctly to the larger system matrix:

            cell->get_dof_indices(local_dof_indices);
            for (unsigned int k = 0; k < n_experiments; ++k)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        system_matrix.add(local_dof_indices[i],
                                          local_dof_indices[j],
                                          local_matrix(i, j));
                    }
                }
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    system_rhs(local_dof_indices[i]) += local_rhs(i);
            }
        }
    }

    /**
     * The run function simply calls the private member functions of the HelmholtzSolver class in order.
     */

    template <int dim>
    void HelmholtzSolver<dim>::run()
    {
        make_grid_and_dofs();
        assemble_system();
        //solve();
        //output_results();
    }

}

// In the main function, we set our parameter values and instantiate an object of type HelmholtzSolver using
// these parameters. All that's left is to call the run function for this object. We enclose all of this in
// a try/catch block in order to better diagnose any runtime errors.

int main()
{
    clock_t start = clock();
    try
    {
        clock_t start = clock();
        using namespace InverseAcoustic;
        const unsigned int fe_degree = 1;
        const unsigned int n_frequencies = 2;
        const unsigned int n_transducers = 2;
        const int regularization_parameter = 0.3;
        HelmholtzSolver<2> helmholtz(fe_degree, n_frequencies, n_transducers, regularization_parameter);
        helmholtz.run();
        clock_t stop = clock();
        double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
        printf("\nTime elapsed: %.5f\n", elapsed);
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}
