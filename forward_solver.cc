/* -----------------------------------------------------------------------
 * This program generates data for use in an inverse acoustic scattering
 * problem. It accomplishes this by solving the Helmholtz equation a total
 * of N = K*F times where K is the number of transducers used and F is
 * the number of frequencies used. The result of the program is N vectors
 * of pressure data where each entry of the vectors is the measurement on
 * one of the K transducers.
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

// The following four include files we need for the treatment of boundary
// values:

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

// This group of files is for the linear algebra which we employ to
// solve the system of equations arising from the finite element discretization
// of the Helmholtz equation. The first two files are for dealing with
// the block structure that arises from the discretization in our problem. The next
// four files are for the local contributions to the block matrix. The final four
// files are used to find inverse operators in the Schur complement and take products
// of operators with vectors.

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

// We need cmath for the value of pi:

#include <cmath>

// This include file allows the program to be timed:

#include <ctime>

// Finally, these files are for output to a file and the console:
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

/**
 * The primary namespace for the forward solver. Helmholtz contains all other namespaces,
 * classes, and functions needed to solve the forward acoustic
 * scattering problem. The main components of Helmholtz are the ForwardSolver
 * class and the PrescribedSolution namespace. This class and namespace
 * contain all information necessary for constructing and solving the forward problem.
 */

namespace Helmholtz
{

    // We include the namespace dealii:

    using namespace dealii;

    /**
    * This class defines all variables and functions needed to assemble
    * and solve the forward acoustic scattering problem.
    */

    template <int dim>
    class ForwardSolver
    {
    public:
        ForwardSolver(const unsigned int degree, const unsigned int n_frequencies, const unsigned int n_transducers);
        void run();

    private:
        void make_grid_and_dofs();
        void assemble_system(unsigned int k);
        void solve(unsigned int k);
        void output_results(unsigned int k);

        // The following constants contain the dimension of the space in
        // which the domain lives, the base degree
        // of the finite element spaces, the number of frequencies
        // transmitted, the number of transducers used to transmit
        // and measure signals, and the number of experiments (which is
        // the number of frequencies multiplied by the number of
        // transducers):

        const unsigned int degree; //!< Degree of freedom for the state variables
        const unsigned int n_frequencies; //!< Number of frequencies transmitted.
        const unsigned int n_transducers; //!< Number of transducers used to transmit and measure signals.
        const unsigned int n_experiments; //!< Total number of experiments (n_frequencies*n_transducers).

        // The following variable is a vector of size n_transducers which
        // will store the location of each of the transducers on the boundary:

        std::vector<types::boundary_id> source_boundary_indicators;

        // The following three objects are used in the
        // construction of our mesh and distribution of DOFs
        // on that mesh:

        Triangulation<dim> triangulation; //!< Describes how we will refine the domain.
        FESystem<dim> fe; //!< Describes the kind of shape functions we want to use.
        DoFHandler<dim> dof_handler;

        // The next four member variables will be used
        // to set the matrix that forms the linear system
        // for our problem:

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;
    };

    /**
     * This namespace assigns the right hand side of the Helmholtz equation and
     * the initial pressure values on the boundary.
     */

    namespace PrescribedSolution
    {

        /**
         * This class sets assigns the right hand side of the Helmholtz equation. RightHandSide
         * inherits from the Function<dim> class from the deal.ii library. It contains
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
         * inherits from the Function<dim> class from the deal.ii library.
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

        // Temporarily set the boundary values on the pressure to 10 Pascals, I want to make
        // sure I understand this function template:

        template <int dim>
        double PressureBoundaryValues<dim>::value(const Point<dim> &p,
                                                  const unsigned int /*component*/) const
        {
            return 10;
        };

        // The following template function sets the value for the parameter Gamma. Gamma is equal
        // to one over the sound speed at every point in the domain. Thus, we will set the value of the
        // sound speed first at each point and then return its inverse to obtain the value of Gamma.

        // I want to look at this. I think not inheriting from Function is fine in this case, but I want
        // to double check a few things. Where is the origin? The center of my disk? If I wanted to add
        // another inhomogeneity offset from the center of the disk, how could I do that?

        template <int dim>
        double Gamma(const Point<dim> &p)
        {
            if (p.square() < 0.5 * 0.5)
            {
                double c = 1700;
                return 1/c;
            }
            else
            {
                double c = 1300;
                return 1/c;
            }
        }
    }

    /**
     * The constructor for the ForwardSolver class, this function takes three parameters and instantiates
     * their corresponding private member variables. After obtaining n_frequencies and n_transducers, it sets
     * the member variable n_experiments to be the product of these two quantities. In addition, the constructor
     * assigns the finite element, the number of refinements performed on the grid, and instantiates the dof handler
     * with the triangulation over the mesh. When assigning fe, we must assign a finite element for each of the
     * state variables. We assign all the state variables to have a finite element of degree
     * "degree".
     * @param degree
     * @param n_frequencies
     * @param n_transducers
     * @param regularization_parameter
     */

    template <int dim>
    ForwardSolver<dim>::ForwardSolver(const unsigned int degree, const unsigned int n_frequencies,
                                          const unsigned int n_transducers, std::vector<types::boundary_id> source_boundary_indicators)
            : degree(degree)
            , n_frequencies(n_frequencies)
            , n_transducers(n_transducers)
            , n_experiments(n_frequencies * n_transducers)
            , fe(FE_Q<dim>(degree), n_experiments)
            , dof_handler(triangulation)
            , source_boundary_indicators.resize(n_transducers)
            , for (unsigned int i=0; i<n_transducers; ++i)
                source_boundary_indicators[i] = 100+i;
    {}

    /**
     * This member function creates the domain, the mesh over that domain, and assigns degrees of freedom
     * to the corresponding mesh. The domain is set to be the unit disc and is refined using the refinement
     * steps set in the constructor for the ForwardSolver class. The degrees of freedom are then assigned
     * to this mesh and the sparsity pattern of the resulting block matrix is assigned.
     */

    template <int dim>
    void ForwardSolver<dim>::make_grid_and_dofs()
    {
        GridGenerator::hyper_ball(triangulation);
        triangulation.refine_global(2);
        dof_handler.distribute_dofs(fe);

        // We will now set the location of the transducers for the problem. First,
        // create a vector of size n_transducers which will hold the point location of
        // each of the centers.

        std::vector<Point<2,double>> transducer_centers(n_transducers);
        for (unsigned int i=0; i<n_transducers; ++i)
        {
            transducer_centers[i][0] = cos((2*M_PI*(i-1)/n_transducers));
            transducer_centers[i][1] = sin((2*M_PI*(i-1)/n_transducers));
        }

        // Now we'll set the radius of each of the transducers:

        const double transducer_radii = 0.2; // I'll need to check the units here to make sure this makes physical sense

        for (auto &cell : triangulation.active_cell_iterators())
        {
            for (auto &face : cell->face_iterators())
            {
                if (face->at_boundary())
                {
                    for (unsigned int i=0; i<n_transducers; ++i)
                    {
                        if (face->center().distance(transducer_centers[i]) < transducer_radii)
                        {
                            face->set_boundary_id(source_boundary_indicators[i]);
                        }
                    }
                }
            }
        }

        // why do we globally refine again?
        triangulation.refine_global(1);

        // Do I need these lines for the forward solver?
        DoFRenumbering::component_wise(dof_handler);
        const std::vector<types::global_dof_index> dofs_per_component =
                DoFTools::count_dofs_per_fe_component(dof_handler);

        // Now figure out how many degrees of freedom are associated with
        // each variable in our solution vector. Each variable in the solution
        // vector represents a pressure measurement corresponding to a different
        // experiment number. Since we set the finite element to be the same
        // for each experiment, the degrees of freedom should be the same
        // for each variable in our solution vector.

        //const unsigned int dof_per_experiment = dofs_per_component[0];

        // Print the number of active cells at each refinement state as well as
        // the total number of degrees of freedom and number of degrees of
        // freedom for each of the state variables:

        std::cout << "   Number of active cells: " << triangulation.n_active_cells()
                  << std::endl;
        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                  << std::endl;

        // We now assign the sparsity pattern for our matrix matrix based
        // on the degrees of freedom:

        DynamicSparsityPattern dsp(n_experiments, n_experiments);
        for (unsigned int block_i=0; block_i<n_experiments; ++block_i)
        {
            for (unsigned int block_j=0; block_j<n_experiments; ++block_j)
                dsp.block(block_i, block_j).reinit(dof_per_experiment, dof_per_experiment);
        }

        // Use the compressed block sparsity pattern to create
        // the sparsity pattern for the full system matrix:

        dsp.collect_sizes();
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);

        // Print the sparsity pattern as a .gnu file:

        std::ofstream out("sparsity_pattern1.gnu");
        sparsity_pattern.print_gnuplot(out);

        // Finally resize the solution and right hand side vectors in exactly the
        // same way as the block compressed sparsity pattern:

        solution.reinit(n_experiments);
        for (unsigned int block_i=0; block_i<n_experiments; ++block_i)
            solution.block(block_i).reinit(dof_per_experiment);
        solution.collect_sizes();

        system_rhs.reinit(n_experiments);
        for (unsigned int block_i=0; block_i<n_experiments; ++block_i)
            system_rhs.block(block_i).reinit(dof_per_experiment);
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
    void ForwardSolver<dim>::assemble_system()
    {
        QGauss<dim> quadrature_formula(degree + 1); // what degree should I choose here?
        QGauss<dim-1> face_quadrature_formula(degree + 1);

        // Specify the quantities we want to update for the shape functions in our
        // finite elements. In this case, we want to update... what here?

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

        // Fill the vectors state_variables with their
        // corresponding finite element indices. This is simply a way for us
        // to label the finite elements in a way that they align with our
        // understanding of the state variable values:

        std::vector<FEValuesExtractors::Scalar> state_variables(n_experiments);
        for (unsigned int i = 0; i < n_experiments; ++i)
            state_variables[i] = FEValuesExtractors::Scalar(i);

        // We need to establish a vector which holds the value of the parameter omega at
        // each experiment. Omega has the value 2*pi*f^k at each experiment k. The central
        // frequency we will use is 150kHZ and we will then use frequencies around 150kHZ in
        // increments of 50kHZ.

        std::vector<double> omega(n_experiments);
        for (unsigned int i=0; i < n_transducers; ++i)
        {
            for (unsigned int j=0; j < n_frequencies; ++j)
            {
                double curr_freq = 150000;
                if (j%2 == 0)
                    curr_freq = 150000 + 50*j;
                else
                    curr_freq = 150000 - 50*j;
                omega[i*n_frequencies + j] = 2*M_PI*curr_freq;
            }
        }

        // Loop over all active cells:

        for (const auto &cell : dof_handler.active_cell_iterators())
        {

            // Reinitialize the local matrices on each cell:

            fe_values.reinit(cell);
            local_matrix = 0;
            local_rhs = 0;
            right_hand_side.value_list(fe_values.get_quadrature_points(),
                                       rhs_values);


            // Loop over the total number of experiments:

            for (unsigned int k = 0; k < n_experiments; ++k)
            {

                // Assign the current value of the parameter omega:

                const double omega_curr = omega[k];

                // Loop over all quadrature points:

                for (unsigned int q = 0; q < n_q_points; ++q)
                {

                    const double gamma_curr = PrescribedSolution::Gamma(fe_values.quadrature_point(q));
                    // Loop over all shape function indices i and j:

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {

                        // Label relevant shape function values:

                        const double phi_i_s = fe_values[state_variables[k]].value(i,q);
                        const Tensor<1, dim> grad_phi_i_s = fe_values[state_variables[k]].gradient(i, q);

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            const double phi_j_s = fe_values[state_variables[k]].value(j, q);
                            const Tensor<1, dim> grad_phi_j_s = fe_values[state_variables[k]].gradient(j, q);

                            // Add the contributions from the bilinear form to our local matrix:

                            local_matrix(i, j) +=
                                    (grad_phi_i_s * grad_phi_j_s -
                                    phi_i_s * gamma_curr * omega_curr * omega_curr * phi_j_s)
                                    *fe_values.JxW(q);
                        }

                        // Add the contributions from the bilinear form to our local right hand side:

                        local_rhs(i) += (phi_i_s * rhs_values[q])
                                        *fe_values.JxW(q);
                    }
                }
            }

            // Keep track of degree of freedom indices on each cell and add these
            // local contributions correctly to the larger system matrix:

            // Do I need the outer loop over k below?
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

        // Now we must add the contribution of the boundary values. The boundary values
        // will change at each experiment and are dependent on the number of transducers and number
        // of frequencies inputted by the user. We will thus loop over the number of transducers
        // and activate each transducer n_frequencies times:

        /*for (unsigned int i=0; i<n_transducers; ++i)
        {
            for (unsigned int j=0; j<n_frequencies; ++j)
            {
                std::map<types::global_dof_index, double> boundary_values;
                VectorTools::interpolate_boundary_values(dof_handler,
                                                         source_boundary_indicators[i],
                                                         PrescribedSolution::PressureBoundaryValues<dim>(),
                                                         boundary_values);
                MatrixTools::apply_boundary_values(boundary_values,
                                                   system_matrix,
                                                   solution,
                                                   system_rhs);
            }
        }*/
    };

    /**
     * The solve function is tasked with solving the linear system assembled and filled in the
     * make_grid_and_dofs() and assemble_system() methods.
     */

    template <int dim>
    void ForwardSolver<dim>::solve()
    {
        // I'll want to discuss the method for solving this problem. We'll still want to take advantage
        // of the block structure, but I'm not sure how in this case.

        // Make a vector of pointers to sparse matrices and two vectors of pointers to vectors. The
        // vector of matrices will hold the locations of the sparse matrices and the vectors of
        // vectors will hold the solutions (pressures) and right hand sides at each experiment. We will
        // fill the elements these point to with the individual block matrices and then carry out
        // each inversion separately:
        std::vector<const SparseMatrix<double> *> M(n_experiments);
        std::vector<Vector<double> *> P(n_experiments);
        std::vector<Vector<double> *> rhs(n_experiments);

        for (unsigned int k=0; k<n_experiments; ++k)
        {
            M[k] = &system_matrix.block(k, k);
            P[k] = &solution.block(k);
            rhs[k] = &system_rhs.block(k);
            SolverControl solver_control(1000, 1e-12);
            SolverCG<Vector<double>> solver(solver_control);
            solver.solve(*M[k], *P[k], *rhs[k], PreconditionIdentity());
        }
    }

    /**
     * The final method in the ForwardSolver class, output_results() takes the pressure values from
     * each experiment and stores them in a vtk file. These pressure measurements will be used in the
     * solution of the inverse problem.
     */

    template <int dim>
    void ForwardSolver<dim>::output_results()
    {
        std::vector<std::string> solution_names;
        for (unsigned int k=0; k<n_experiments; ++k)
            solution_names.emplace_back("pressure_" + std::to_string(k));
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
                interpretation(n_experiments,
                               DataComponentInterpretation::component_is_scalar);
        DataOut<dim> data_out;
        data_out.add_data_vector(dof_handler,
                                 solution,
                                 solution_names,
                                 interpretation);
        data_out.build_patches(degree + 1); // should I include this in my program?
        std::ofstream output("solution.vtk");
        data_out.write_vtk(output);
    }

    /**
     * The run function simply calls the private member functions of the HelmholtzSolver class in order.
     */

    template <int dim>
    void ForwardSolver<dim>::run()
    {
        make_grid_and_dofs();

        for (unsigned int k=0; k<n_experiments; ++k)
        {
            assemble_system(k);
            solve(k);
            output_results(k);
        }
    }

};

// In the main function, we set our parameter values and instantiate an object of type HelmholtzSolver using
// these parameters. All that's left is to call the run function for this object. We enclose all of this in
// a try/catch block in order to better diagnose any runtime errors.

int main() {
    clock_t start = clock();
    try {
        clock_t start = clock();
        using namespace Helmholtz;
        const unsigned int fe_degree = 1;
        const unsigned int n_frequencies = 3;
        const unsigned int n_transducers = 4;
        ForwardSolver<2> helmholtz(fe_degree, n_frequencies, n_transducers);
        helmholtz.run();
        clock_t stop = clock();
        double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
        printf("\nTime elapsed: %.5f\n", elapsed);
    }
    catch (std::exception &exc) {
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
    catch (...) {
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