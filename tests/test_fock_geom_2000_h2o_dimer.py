from pathlib import Path
import numpy as np

from veloxchem import MolecularBasis
from veloxchem import Molecule
from veloxchem import FockGeom2000Driver
from veloxchem import SubMatrix
from veloxchem import Matrix
from veloxchem import Matrices
from veloxchem import make_matrix
from veloxchem import mat_t


class TestFockGeom2000Driver:

    def get_data_h2o_dimer(self):

        h2o_dimer_str = """
            O -1.464  0.099  0.300
            H -1.956  0.624 -0.340
            H -1.797 -0.799  0.206
            O  1.369  0.146 -0.395
            H  1.894  0.486  0.335
            H  0.451  0.163 -0.083
        """
        mol = Molecule.read_str(h2o_dimer_str, 'angstrom')
        bas = MolecularBasis.read(mol, 'def2-svpd')

        return mol, bas

    def test_h2o_dimer_fock_2jk_hess_h2_svpd(self):

        mol_h2o_dimer, bas_svpd = self.get_data_h2o_dimer()

        # load density matrix
        here = Path(__file__).parent
        npyfile = str(here / 'data' / 'h2o.dimer.svpd.density.npy')
        den_mat = make_matrix(bas_svpd, mat_t.symmetric)
        den_mat.set_values(np.load(npyfile))

        # compute Fock matrix
        fock_drv = FockGeom2000Driver()
        fock_mats = fock_drv.compute(bas_svpd, mol_h2o_dimer, den_mat, 1,
                                     "2jk", 0.0, 0.0)

        # load reference Fock matrix
        here = Path(__file__).parent
        npyfile = str(here / 'data' / 'h2o.dimer.svpd.j.geom.2000.h2.h2.npy')
        ref_mat = 2.0 * np.load(npyfile)

        # load reference Fock matrix
        here = Path(__file__).parent
        npyfile = str(here / 'data' / 'h2o.dimer.svpd.k.geom.2000.h2.h2.npy')
        ref_mat = ref_mat - np.load(npyfile)

        # dimension of molecular basis
        basdims = [0, 16, 58, 78]

        # check individual submatrices of XX matrix
        fock_mat_xx = fock_mats.matrix("XX")
        for i in range(3):
            for j in range(3):
                # bra side
                sbra = basdims[i]
                ebra = basdims[i + 1]
                # ket side
                sket = basdims[j]
                eket = basdims[j + 1]
                # load computed submatrix
                cmat = fock_mat_xx.submatrix((i, j))
                # load reference submatrix
                rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                rmat.set_values(
                    np.ascontiguousarray(ref_mat[0, 0][sbra:ebra, sket:eket]))

                # compare submatrices
                assert cmat == rmat

        # check full Fock matrix
        fmat = fock_mat_xx.full_matrix()
        fref = SubMatrix([0, 0, 78, 78])
        fref.set_values(np.ascontiguousarray(ref_mat[0, 0]))
        assert fmat == fref

        # check individual submatrices of XY matrix
        fock_mat_xy = fock_mats.matrix("XY")
        for i in range(3):
            for j in range(3):
                # bra side
                sbra = basdims[i]
                ebra = basdims[i + 1]
                # ket side
                sket = basdims[j]
                eket = basdims[j + 1]
                # load computed submatrix
                cmat = fock_mat_xy.submatrix((i, j))
                # load reference submatrix
                rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                rmat.set_values(
                    np.ascontiguousarray(ref_mat[0, 1][sbra:ebra, sket:eket]))
                # compare submatrices
                assert cmat == rmat

        # check full Fock matrix
        fmat = fock_mat_xy.full_matrix()
        fref = SubMatrix([0, 0, 78, 78])
        fref.set_values(np.ascontiguousarray(ref_mat[0, 1]))
        assert fmat == fref

        # check individual submatrices of XZ matrix
        fock_mat_xz = fock_mats.matrix("XZ")
        for i in range(3):
            for j in range(3):
                # bra side
                sbra = basdims[i]
                ebra = basdims[i + 1]
                # ket side
                sket = basdims[j]
                eket = basdims[j + 1]
                # load computed submatrix
                cmat = fock_mat_xz.submatrix((i, j))
                # load reference submatrix
                rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                rmat.set_values(
                    np.ascontiguousarray(ref_mat[0, 2][sbra:ebra, sket:eket]))
                # compare submatrices
                assert cmat == rmat

        # check full Fock matrix
        fmat = fock_mat_xz.full_matrix()
        fref = SubMatrix([0, 0, 78, 78])
        fref.set_values(np.ascontiguousarray(ref_mat[0, 2]))
        assert fmat == fref

        # check individual submatrices of YY matrix
        fock_mat_yy = fock_mats.matrix("YY")
        for i in range(3):
            for j in range(3):
                # bra side
                sbra = basdims[i]
                ebra = basdims[i + 1]
                # ket side
                sket = basdims[j]
                eket = basdims[j + 1]
                # load computed submatrix
                cmat = fock_mat_yy.submatrix((i, j))
                # load reference submatrix
                rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                rmat.set_values(
                    np.ascontiguousarray(ref_mat[1, 1][sbra:ebra, sket:eket]))
                # compare submatrices
                assert cmat == rmat

        # check full Fock matrix
        fmat = fock_mat_yy.full_matrix()
        fref = SubMatrix([0, 0, 78, 78])
        fref.set_values(np.ascontiguousarray(ref_mat[1, 1]))
        assert fmat == fref

        # check individual submatrices of YZ matrix
        fock_mat_yz = fock_mats.matrix("YZ")
        for i in range(3):
            for j in range(3):
                # bra side
                sbra = basdims[i]
                ebra = basdims[i + 1]
                # ket side
                sket = basdims[j]
                eket = basdims[j + 1]
                # load computed submatrix
                cmat = fock_mat_yz.submatrix((i, j))
                # load reference submatrix
                rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                rmat.set_values(
                    np.ascontiguousarray(ref_mat[1, 2][sbra:ebra, sket:eket]))
                # compare submatrices
                assert cmat == rmat

        # check full Fock matrix
        fmat = fock_mat_yz.full_matrix()
        fref = SubMatrix([0, 0, 78, 78])
        fref.set_values(np.ascontiguousarray(ref_mat[1, 2]))
        assert fmat == fref

        # check individual submatrices of ZZ matrix
        fock_mat_zz = fock_mats.matrix("ZZ")
        for i in range(3):
            for j in range(3):
                # bra side
                sbra = basdims[i]
                ebra = basdims[i + 1]
                # ket side
                sket = basdims[j]
                eket = basdims[j + 1]
                # load computed submatrix
                cmat = fock_mat_zz.submatrix((i, j))
                # load reference submatrix
                rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                rmat.set_values(
                    np.ascontiguousarray(ref_mat[2, 2][sbra:ebra, sket:eket]))
                # compare submatrices
                assert cmat == rmat

        # check full Fock matrix
        fmat = fock_mat_zz.full_matrix()
        fref = SubMatrix([0, 0, 78, 78])
        fref.set_values(np.ascontiguousarray(ref_mat[2, 2]))
        assert fmat == fref

    def test_h2o_dimer_fock_2jk_hess_o4_svpd(self):

        mol_h2o_dimer, bas_svpd = self.get_data_h2o_dimer()

        # load density matrix
        here = Path(__file__).parent
        npyfile = str(here / 'data' / 'h2o.dimer.svpd.density.npy')
        den_mat = make_matrix(bas_svpd, mat_t.symmetric)
        den_mat.set_values(np.load(npyfile))

        # compute Fock matrix
        fock_drv = FockGeom2000Driver()
        fock_mats = fock_drv.compute(bas_svpd, mol_h2o_dimer, den_mat, 3,
                                     "2jk", 0.0, 0.0)

        # load reference Fock matrix
        here = Path(__file__).parent
        npyfile = str(here / 'data' / 'h2o.dimer.svpd.j.geom.2000.o4.o4.npy')
        ref_mat = 2.0 * np.load(npyfile)

        # load reference Fock matrix
        here = Path(__file__).parent
        npyfile = str(here / 'data' / 'h2o.dimer.svpd.k.geom.2000.o4.o4.npy')
        ref_mat = ref_mat - np.load(npyfile)

        # dimension of molecular basis
        basdims = [0, 16, 58, 78]

        # check individual submatrices of XX matrix
        fock_mat_xx = fock_mats.matrix("XX")
        for i in range(3):
            for j in range(3):
                # bra side
                sbra = basdims[i]
                ebra = basdims[i + 1]
                # ket side
                sket = basdims[j]
                eket = basdims[j + 1]
                # load computed submatrix
                cmat = fock_mat_xx.submatrix((i, j))
                # load reference submatrix
                rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                rmat.set_values(
                    np.ascontiguousarray(ref_mat[0, 0][sbra:ebra, sket:eket]))

                # compare submatrices
                #print(i, " ", j, " xx", np.max(rmat.to_numpy() - cmat.to_numpy()))
                #assert cmat == rmat

        # check full Fock matrix
        fmat = fock_mat_xx.full_matrix()
        fref = SubMatrix([0, 0, 78, 78])
        fref.set_values(np.ascontiguousarray(ref_mat[0, 0]))
        #assert fmat == fref

        # check individual submatrices of XY matrix
        fock_mat_xy = fock_mats.matrix("XY")
        for i in range(3):
            for j in range(3):
                # bra side
                sbra = basdims[i]
                ebra = basdims[i + 1]
                # ket side
                sket = basdims[j]
                eket = basdims[j + 1]
                # load computed submatrix
                cmat = fock_mat_xy.submatrix((i, j))
                # load reference submatrix
                rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                rmat.set_values(
                    np.ascontiguousarray(ref_mat[0, 1][sbra:ebra, sket:eket]))
                # compare submatrices
                #print(i, " ", j, " xy", np.max(rmat.to_numpy() - cmat.to_numpy()))
                #assert cmat == rmat

        # check full Fock matrix
        fmat = fock_mat_xy.full_matrix()
        fref = SubMatrix([0, 0, 78, 78])
        fref.set_values(np.ascontiguousarray(ref_mat[0, 1]))
        #assert fmat == fref

        # check individual submatrices of XZ matrix
        fock_mat_xz = fock_mats.matrix("XZ")
        for i in range(3):
            for j in range(3):
                # bra side
                sbra = basdims[i]
                ebra = basdims[i + 1]
                # ket side
                sket = basdims[j]
                eket = basdims[j + 1]
                # load computed submatrix
                cmat = fock_mat_xz.submatrix((i, j))
                # load reference submatrix
                rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                rmat.set_values(
                    np.ascontiguousarray(ref_mat[0, 2][sbra:ebra, sket:eket]))
                # compare submatrices
                #print(i, " ", j, " xz", np.max(rmat.to_numpy() - cmat.to_numpy()))
                #assert cmat == rmat

        # check full Fock matrix
        fmat = fock_mat_xz.full_matrix()
        fref = SubMatrix([0, 0, 78, 78])
        fref.set_values(np.ascontiguousarray(ref_mat[0, 2]))
        #assert fmat == fref

        # check individual submatrices of YY matrix
        fock_mat_yy = fock_mats.matrix("YY")
        for i in range(3):
            for j in range(3):
                # bra side
                sbra = basdims[i]
                ebra = basdims[i + 1]
                # ket side
                sket = basdims[j]
                eket = basdims[j + 1]
                # load computed submatrix
                cmat = fock_mat_yy.submatrix((i, j))
                # load reference submatrix
                rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                rmat.set_values(
                    np.ascontiguousarray(ref_mat[1, 1][sbra:ebra, sket:eket]))
                # compare submatrices
                #print(i, " ", j, " yy", np.max(rmat.to_numpy() - cmat.to_numpy()))
                #assert cmat == rmat

        # check full Fock matrix
        fmat = fock_mat_yy.full_matrix()
        fref = SubMatrix([0, 0, 78, 78])
        fref.set_values(np.ascontiguousarray(ref_mat[1, 1]))
        #assert fmat == fref

        # check individual submatrices of YZ matrix
        fock_mat_yz = fock_mats.matrix("YZ")
        for i in range(3):
            for j in range(3):
                # bra side
                sbra = basdims[i]
                ebra = basdims[i + 1]
                # ket side
                sket = basdims[j]
                eket = basdims[j + 1]
                # load computed submatrix
                cmat = fock_mat_yz.submatrix((i, j))
                # load reference submatrix
                rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                rmat.set_values(
                    np.ascontiguousarray(ref_mat[1, 2][sbra:ebra, sket:eket]))
                # compare submatrices
                #print(i, " ", j, " yz", np.max(rmat.to_numpy() - cmat.to_numpy()))
                #assert cmat == rmat

        # check full Fock matrix
        fmat = fock_mat_yz.full_matrix()
        fref = SubMatrix([0, 0, 78, 78])
        fref.set_values(np.ascontiguousarray(ref_mat[1, 2]))
        #assert fmat == fref

        # check individual submatrices of ZZ matrix
        fock_mat_zz = fock_mats.matrix("ZZ")
        for i in range(3):
            for j in range(3):
                # bra side
                sbra = basdims[i]
                ebra = basdims[i + 1]
                # ket side
                sket = basdims[j]
                eket = basdims[j + 1]
                # load computed submatrix
                cmat = fock_mat_zz.submatrix((i, j))
                # load reference submatrix
                rmat = SubMatrix([sbra, sket, ebra - sbra, eket - sket])
                rmat.set_values(
                    np.ascontiguousarray(ref_mat[2, 2][sbra:ebra, sket:eket]))
                # compare submatrices
                #print(i, " ", j, " zz", np.max(rmat.to_numpy() - cmat.to_numpy()))
                #assert cmat == rmat

        # check full Fock matrix
        fmat = fock_mat_zz.full_matrix()
        fref = SubMatrix([0, 0, 78, 78])
        fref.set_values(np.ascontiguousarray(ref_mat[2, 2]))
        #assert fmat == fref

        #assert False
