/* ---------------------------------------------------------------------
 * This program builds a block matrix that corresponds to one Newton
 * step in an inverse scattering problem (in 2D) along with the
 * corresponding right hand side. The code from step-22 of the deal.ii
 * software package is used as a foundation for this program.
 * ---------------------------------------------------------------------

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

// The next three files contain classes which are needed for loops over all
// cells and to get the information from the cell objects. The first two are
// used to get geometric information from cells; the last one provides
// information about the degrees of freedom local to a cell:
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

// This file contains the description of the Lagrange interpolation finite
// element:
#include <deal.II/fe/fe_q.h>

// And these files are needed for the creation of sparsity patterns of sparse
// matrices and renumbering of degrees of freedom:
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

// The next four files are needed for assembling the block matrix using quadrature on
// each cell:
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

// The following three include files we need for the treatment of boundary
// values:
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

// This group of include files is for the linear algebra which we employ to
// solve the system of equations arising from the finite element discretization
// of the Helmholtz equation. The first three packages are for dealing with
// the block structure that arises from the discretization in our problem.
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
//#include <deal.II/lac/solver_cg.h>
//#include <deal.II/lac/precondition.h>

// Finally, this is for output to a file and to the console:
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

// We include the namespace dealii
using namespace dealii;

// The main class for this problem is HelmholtzSolver. This
// class has two public functions, a constructor and a run function.
// Later, in the main() function, we will simply need to call the
// constructor and run function. The private member functions are
// make_grid will all be called in the run() function and are explained
// below in their definitions.

class HelmholtzSolver
{
public:

    HelmholtzSolver(const unsigned int degree);

    void run();

private:

    void make_grid_and_dofs();
    //void assemble_system();

    const unsigned int degree;

    Triangulation<2> triangulation;
    FESystem<2>      fe;
    DoFHandler<2>    dof_handler;

    //BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    //BlockVector<double> solution;
    //BlockVector<double> system_rhs;
};

// Here we define the constructor for our HelmholtzSolver class. It specifies
// that we want bi-linear elements (denoted by the parameter to the finite
// element (fe) object, which indicates the polynomial degree), and to associate
// the dof_handler variable to the triangulation we use.
HelmholtzSolver::HelmholtzSolver(const unsigned int degree)
    : degree(degree)
    , fe(FE_Q<2>(degree + 1), 2, FE_Q<2>(degree), 1)
    , dof_handler(triangulation)
{}

// Next we define the private member function make_grid(), which
// constructs the grid on which we will solve our problem. We use the
// dealii function GridGenerator::hyper_ball, which produces a triangulation
// over the unit disc in 2D space centered at the origin. We globally
// refine this triangulation twice and then print the ouput as a .svg file
// under the name helmholtz_grid_1.svg.

void HelmholtzSolver::make_grid_and_dofs()
{
    GridGenerator::hyper_ball(triangulation);
    triangulation.refine_global(2);

    std::ofstream out("helmholtz_grid_1.svg");
    GridOut       grid_out;
    grid_out.write_svg(triangulation, out);
    std::cout << "Grid written to helmholtz_grid_1.svg" << std::endl;

    // We then associate degrees of freedom
    // with this mesh.

    dof_handler.distribute_dofs(fe);

    // Now we want to subdivide our matrix into block corresponding
    // to the directions for pressures, adjoint variables, and
    // parameter q.

    system_matrix.clear();

    // We will want to renumber the degrees of freedom on our mesh
    // in such a way to minimize the bandwidth of our matrix. This
    // is accomplished through the following function Cuthill_Mckee.

    DoFRenumbering::Cuthill_McKee(dof_handler);
    std::vector<unsigned int> block_component(3, 0);
}

// @sect4{Step3::run}

// The last function of this class is the main function which calls
// all the other functions of the <code>HelmholtzSolver</code> class.
// The order in which this is done resembles the order in which most
// finite element programs work.

void HelmholtzSolver::run()
{
    make_grid_and_dofs();
    //setup_system();
    //assemble_system();
    //solve();
    //output_results();
}

int main()
{
  HelmholtzSolver helmholtz(1);
  helmholtz.run();

  return 0;
}
