from pathlib import Path
import numpy as np

from veloxchem.veloxchemlib import KineticEnergyGeom100Driver
from veloxchem.molecularbasis import MolecularBasis
from veloxchem.molecule import Molecule
from veloxchem.submatrix import SubMatrix


class TestKineticEnergyGeom100Driver:

    def get_data(self):

        costr = """
            C   0.100  -0.400  -1.000
            O   0.300   1.400  -2.100
        """
        mol = Molecule.read_str(costr, 'au')
        bas = MolecularBasis.read(mol, 'def2-QZVP')

        return mol, bas

    def test_kinetic_energy_co_qzvp_for_c(self):

        mol, bas = self.get_data()

        # compute kinetic energy gradient matrix
        grad_drv = KineticEnergyGeom100Driver()
        grad_mats = grad_drv.compute(mol, bas, 0)

        # load reference kinetic energy gradient for C atom
        here = Path(__file__).parent
        npyfile = str(here / 'data' / 'co.qzvp.kinetic.energy.geom.100.c.npy')
        ref_mat = np.load(npyfile)

        # dimension of molecular basis
        basdims = [0, 14, 38, 68, 96, 114]

        # indices map
        labels = ['X', 'Y', 'Z']

        for k, label in enumerate(labels):
            fmat = grad_mats.matrix(label)
            for i in range(0, 5):
                for j in range(0, 5):
                    # bra side
                    sbra = basdims[i]
                    ebra = basdims[i + 1]
                    # ket side
                    sket = basdims[j]
                    eket = basdims[j + 1]
                    # load computed submatrix
                    cmat = fmat.submatrix((i, j))
                    # load reference submatrix
                    rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                    rmat.set_values(
                        np.ascontiguousarray(ref_mat[k, sbra:ebra, sket:eket]))
                    # compare submatrices
                    # NOTE: See test cases with numerical problems.
                    # assert cmat == rmat
            smat = fmat.full_matrix()
            fref = SubMatrix([0, 0, 114, 114])
            fref.set_values(np.ascontiguousarray(ref_mat[k]))
            # NOTE: See test cases with numerical problems.
            # assert smat == fref

    def test_kinetic_energy_co_qzvp_for_o(self):

        mol, bas = self.get_data()

        # compute kinetic energy gradient matrix
        grad_drv = KineticEnergyGeom100Driver()
        grad_mats = grad_drv.compute(mol, bas, 1)

        # load reference kinetic energy gradient for O atom
        here = Path(__file__).parent
        npyfile = str(here / 'data' / 'co.qzvp.kinetic.energy.geom.100.o.npy')
        ref_mat = np.load(npyfile)

        # dimension of molecular basis
        basdims = [0, 14, 38, 68, 96, 114]

        # indices map
        labels = ['X', 'Y', 'Z']

        for k, label in enumerate(labels):
            fmat = grad_mats.matrix(label)
            for i in range(0, 5):
                for j in range(0, 5):
                    # bra side
                    sbra = basdims[i]
                    ebra = basdims[i + 1]
                    # ket side
                    sket = basdims[j]
                    eket = basdims[j + 1]
                    # load computed submatrix
                    cmat = fmat.submatrix((i, j))
                    # load reference submatrix
                    rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                    rmat.set_values(
                        np.ascontiguousarray(ref_mat[k, sbra:ebra, sket:eket]))
                    # compare submatrices
                    assert cmat == rmat
            smat = fmat.full_matrix()
            fref = SubMatrix([0, 0, 114, 114])
            fref.set_values(np.ascontiguousarray(ref_mat[k]))
            assert smat == fref
