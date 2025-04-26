#______________________________________________________________________________
from typing import List, Optional
from itertools import product
from sympy import Matrix, eye, sympify
from gr_tensor_calculator import Coordinate, Basis, Basis_Vector


def build_coords(*, labels: List[str]) -> List[Coordinate]:
    return [Coordinate(i, label) for i, label in enumerate(labels)]


def build_alt_basis(
    *, flat_components: List[str], coords: List[Coordinate], n: int
) -> Basis:
    basis_vectors: List[Basis_Vector] = []
    for i in range(n):
        vec_components = flat_components[i * n:(i + 1) * n]
        vec = Basis_Vector(index=i, use_str=vec_components, coords=coords)
        basis_vectors.append(vec)
    return Basis(basis_vectors)


def build_metric_matrix(
    *,
    component_strings: Optional[List[str]] = None,
    n: int,
    diagonal: Optional[bool] = None,
    use_alt_basis: bool = False,
    alt_basis: Optional[Basis] = None,
    ortho: Optional[bool] = None,
    metric_in_CB: Optional[bool] = None,
    is_pseudo_riemannian: Optional[bool] = None
) -> Matrix:

    if not use_alt_basis:
        if alt_basis is not None:
            raise ValueError(
                "Alternate basis is disabled, but "
                "a Basis object was provided."
            )
        if ortho is not None:
            raise ValueError(
                "Alternate basis is disabled, so "
                "orthonormality does not need to be specified."
            )
        if metric_in_CB is not None:
            raise ValueError(
                "Alternate basis is disabled, so metric "
                "basis does not need to be specified."
            )
        if is_pseudo_riemannian is not None:
            raise ValueError(
                "Alternate basis is disabled, so whether or not manifold "
                "is pseudo-Riemannian does not need to be specified."
            )

    if component_strings is not None:
        components = [sympify(c) for c in component_strings]
        if diagonal:
            pre_g_m = Matrix.diag(*components)
        else:
            pre_g_m = Matrix.zeros(n)
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    pre_g_m[i, j] = pre_g_m[j, i] = components[idx]
                    idx += 1
        if not use_alt_basis or metric_in_CB:
            g_m = pre_g_m

    if use_alt_basis:
        if alt_basis is None:
            raise ValueError(
                "Alternate basis is enabled, "
                "but no Basis object was provided."
            )
        if ortho is None:
            raise ValueError(
                "Alternate basis is enabled, but whether or not "
                "basis is orthonormal is not specified."
            )
        if not ortho:
            if component_strings is None:
                raise ValueError(
                    "Alternate basis is not orthonormal, "
                    "but no metric components were provided."
                )
            if metric_in_CB is None:
                raise ValueError(
                    "Alternate basis is not orthonormal, "
                    "but metric components' basis was not specified."
                )
        if component_strings is None and is_pseudo_riemannian is None:
            raise ValueError(
                "Metric components were not provided, but whether or not "
                "manifold is pseudo-Riemannian is not specified."
            )
        if (
            component_strings is None or 
            (ortho is False and metric_in_CB is False)
        ):
            M = Matrix([
                [k*l for k, l in product(BV1.use, BV2.use)]
                for BV1, BV2 in product(alt_basis.basis_set, repeat=2)
            ])
            if ortho:
                v = eye(n).reshape(n**2, 1)
                if is_pseudo_riemannian:
                    v[0] = -1
            else:
                v = pre_g_m.reshape(n**2, 1)
            g_m = M.LUsolve(v).reshape(n, n)

    if 'g_m' not in locals():
        raise ValueError(
            "Metric could not be constructed from the "
            "provided input."
        )
    return g_m