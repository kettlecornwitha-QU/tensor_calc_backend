#______________________________________________________________________________
from dataclasses import dataclass
from typing import List, Dict
from itertools import product
from sympy import (
    MutableDenseNDimArray as MDMA, Symbol, eye, Matrix, S, diff,
    tensorcontraction as contract, tensorproduct as tensprod, latex,
    Eq, simplify, Expr, sympify
)


def latex_eq(symbol: str, expression: Expr) -> str:
    return latex(Eq(Symbol(symbol), expression))


def var_rank_array(dim: int, rank: int) -> MDMA:
    x = 0
    for i in range(rank):
        x = [x,] * dim
    return MDMA(x)


@dataclass
class Coordinate:
    index: int
    label: str
    
    def __post_init__(self) -> None:
        self.symbol = Symbol(self.label)
        self.latex = latex(self.symbol)
    
    @classmethod
    def from_stdin(cls, index: int) -> 'Coordinate':
        return cls(index, User_Ins.get_coord_label(index))
        
    def __str__(self) -> str:
        return self.label


class Basis:
    def __init__(self, basis_set: list['Basis_Vector']) -> None:
        self.basis_set = basis_set
        self.dual_basis = self.calc_dual_basis()
        
    def calc_dual_basis(self) -> list['Basis_One_Form']:
        dim = len(self.basis_set)
        basis = [self.basis_set[i].use_str for i in range(dim)]
        dual_basis = []
        for i, dual_vector in enumerate(Matrix(basis).T.inv().tolist()):
            coords = self.basis_set[i].coords
            dual_basis.append(
                Basis_One_Form(
                    i, MDMA(
                        [sympify(component) for component in dual_vector]
                    ), coords
                )
            )
        return dual_basis


class GR_Array:
    def __init__(
        self, name: str, symbol: str, key: str, use: MDMA, 
        coords: list[Coordinate], alt_basis: Basis=None
    ) -> None:
        self.name = name
        self.symbol = symbol
        self.key = key
        self.use = use
        self.coords = coords
        self.n = len(coords)
    
    @property
    def rank(self) -> int:
        return self.key.count('*')
    
    def finalize_array(self, using_alt_basis: bool = False) -> List[str]:
        index = [coord.label for coord in self.coords]
        base_key, use = self.key, self.use
        if isinstance(self, Tensor):
            base_key = self.disp_key
            if using_alt_basis:
                index = [f'{i}' for i in range(self.n)]
                use = self.alt_basis_use
        non_zero_components = []
        for i in product(range(self.n), repeat=self.rank):
            if use[i] == 0:
                continue
            index_key = base_key.replace('*', '%s') % tuple(index[j] 
                                                            for j in i)
            non_zero_components.append(
                latex_eq((self.symbol + index_key), simplify(use[i]))
            )
        return non_zero_components

    def partial_derivative(self) -> 'GR_Array':
        new_key = self.key + '_,_*'
        new_use = var_rank_array(self.n, self.rank+1)
        for i in product(range(self.n), repeat=self.rank+1):
            new_use[i] = diff(self.use[i[:-1]], self.coords[i[-1]].symbol)
        return GR_Array(self.name, self.symbol, new_key, new_use, self.coords)


class Tensor(GR_Array):
    def __init__(
        self, name: str, symbol: str, key: str, 
        use: MDMA, coords: List[Coordinate], alt_basis: Basis = None
    ) -> None:
        super().__init__(name, symbol, key, use, coords)
        self.disp_key = self.generate_disp_key()
        self.alt_basis_use = self.change_basis(alt_basis)
        
    def raise_index(
        self, i: int, g_inv: MDMA, alt_basis: Basis
    ) -> 'Tensor':
        if self.key[i*2] != '_':
            if self.key[i*2] == '^':
                raise ValueError('That index is already raised')
            raise ValueError('Your key is not properly formatted')
        new_key = self.key[:i*2] + '^' + self.key[i*2+1:]
        new_use = contract(tensprod(self.use, g_inv), (i, self.rank))
        return Tensor(
            self.name, self.symbol, new_key, 
            new_use, self.coords, alt_basis
        )
    
    def lower_index(self, i: int, g: MDMA, alt_basis: Basis) -> 'Tensor':
        if self.key[i*2] != '^':
            if self.key[i*2] == '_':
                raise ValueError('That index is already lowered')
            raise ValueError('Your key is not properly formatted')
        new_key = self.key[:i*2] + '_' + self.key[i*2+1:]
        new_use = contract(tensprod(self.use, g), (i, self.rank))
        return Tensor(
            self.name, self.symbol, new_key, 
            new_use, self.coords, alt_basis
        )
    
    def change_basis(self, alt_basis: Basis) -> MDMA:
        if alt_basis is None:
            return None
        dim = len(alt_basis.basis_set)
        alt_tensor = var_rank_array(dim, self.rank)
        key = self.key.replace('*', '')
        for i in product(range(dim), repeat=self.rank):
            mixed_basis = []
            for k, l in enumerate(key):
                if l == '_':
                    mixed_basis.append(alt_basis.basis_set[i[k]].use)
                else:
                    mixed_basis.append(alt_basis.dual_basis[i[k]].use)
            temp = contract(tensprod(self.use, mixed_basis[0]), (0, self.rank))
            for j, vect in enumerate(mixed_basis[1:]):
                temp = contract(tensprod(temp, vect), (0, self.rank-1-j))
            alt_tensor[i] = simplify(temp)
        return alt_tensor

    def generate_disp_key(self) -> str:
        output = []
        prev_char = None
        consecutive_count = 0
        for i in range(0, len(self.key), 2):
            current = self.key[i:i+2]
            if prev_char is not None and current != prev_char:
                if prev_char == '^*' and current == '_*':
                    output.append('_~_~' * consecutive_count)
                elif prev_char == '_*' and current == '^*':
                    output.append('^~^~' * consecutive_count)
                consecutive_count = 0
            output.append(current)
            prev_char = current
            consecutive_count += 1
        return ''.join(output)
    
    def finalize_tensor(self) -> List[str]:
        if self.alt_basis_use is None:
            return self.finalize_array()
        return (
            self.finalize_array() + self.finalize_array(using_alt_basis=True)
        )


class Metric(Tensor):
    def __init__(
        self, g_m: Matrix, coords: list[Coordinate], alt_basis: Basis
    ) -> None:
        name = 'metric tensor'
        symbol = 'g'
        key = '_*_*'
        use = MDMA(g_m)
        super().__init__(name, symbol, key, use, coords, alt_basis)
        

class Inverse_Metric(Tensor):
    def __init__(
        self, metric: Metric, coords: list[Coordinate], alt_basis: Basis
    ) -> None:
        name = 'inverse metric'
        symbol = 'g'
        key = '^*^*'
        use = simplify(MDMA(Matrix(metric.use).inv()))
        super().__init__(name, symbol, key, use, coords, alt_basis)


class Basis_Vector(Tensor):
    def __init__(
        self, index: int, use_str: list[str], coords: list[Coordinate]
    ) -> None:
        name = f'basis vector {index}'
        symbol = 'e'
        key = '^*'
        use = MDMA([sympify(i) for i in use_str])
        super().__init__(name, symbol, key, use, coords)
        self.index = index
        self.use_str = use_str
        self.latex = r'\mathbf{e}_{%s}' % index
        

class Basis_One_Form(Tensor):
    def __init__(
        self, index: int, use: MDMA, coords: list[Coordinate]
    ) -> None:
        name = f'basis one-form {index}'
        symbol = 'omega'
        key = '_*'
        super().__init__(name, symbol, key, use, coords)
        self.index = index
        self.latex = r'\mathbf{\omega}_{%s}' % index


class Christoffel(GR_Array):
    def __init__(
        self, coords: list[Coordinate], metric: Metric, 
        inverse_metric: Inverse_Metric
    ) -> None:
        name = 'Christoffel symbol - 2nd kind'
        symbol = 'Gamma'
        key = '^*_*_*'
        use = self.calc_christoffel(metric, inverse_metric, coords)
        super().__init__(name, symbol, key, use, coords)

    def calc_christoffel(
        self, metric: Metric, inverse_metric: Inverse_Metric, 
        coords: list[Coordinate]
    ) -> MDMA:
        n = len(coords)
        g_inv, g_d = inverse_metric.use, metric.partial_derivative().use
        Gamma = MDMA.zeros(n, n, n)
        for i, j, k, l in product(range(n), repeat=4):
            Gamma[i, j, k] += S(1)/2 * g_inv[i, l] * (
                g_d[k, l, j] + g_d[l, j, k] - g_d[j, k, l]
            )
        return Gamma


class Riemann(Tensor):
    def __init__(
        self, christoffel: Christoffel, 
        coords: list[Coordinate], alt_basis: Basis
    ) -> None:
        name = 'Riemann curvature tensor'
        symbol = 'R'
        key = '^*_*_*_*'
        use = self.calc_rie(christoffel, coords)
        super().__init__(name, symbol, key, use, coords, alt_basis)

    def calc_rie(
        self, christoffel: Christoffel, coords: list[Coordinate]
    ) -> MDMA:
        n = len(coords)
        Gamma, Gamma_d = christoffel.use, christoffel.partial_derivative().use
        Rie = MDMA.zeros(n, n, n, n)
        for i, j, k, l in product(range(n), repeat=4):
            Rie[i, j, k, l] = Gamma_d[i, j, l, k] - Gamma_d[i, j, k, l]
            for m in range(n):
                Rie[i, j, k, l] += (
                    Gamma[m, j, l] * Gamma[i, m, k]
                    - Gamma[m, j, k] * Gamma[i, m, l]
                )
            Rie[i, j, k, l] = simplify(Rie[i, j, k, l])
        return Rie


class Ricci_Tensor(Tensor):
    def __init__(
        self, rie: Riemann, coords: list[Coordinate], alt_basis: Basis
    ) -> None:
        name = 'Ricci curvature tensor'
        symbol = 'R'
        key = '_*_*'
        use = simplify(contract(rie.use, (0, 2)))
        super().__init__(name, symbol, key, use, coords, alt_basis)


class Ricci_Scalar:
    def __init__(
        self, ric: Ricci_Tensor, g_inv: MDMA, alt_basis: Basis
    ) -> None:
        self.name = 'Ricci scalar'
        self.symbol = 'R'
        self.value = simplify(
            contract(ric.raise_index(0, g_inv, alt_basis).use, (0, 1))
        )
        self.finalized = latex_eq('R', self.value)


class Einstein(Tensor):
    def __init__(
        self, metric: Metric, ric_t: Ricci_Tensor, ric_s: Ricci_Scalar, 
        coords: list[Coordinate], alt_basis: Basis
    ) -> None:
        name = 'Einstein tensor'
        symbol = 'G'
        key = '_*_*'
        use = self.calc_einstein(metric, ric_t, ric_s, coords)
        super().__init__(name, symbol, key, use, coords, alt_basis)

    def calc_einstein(
        self, metric: Metric, ric_t: Ricci_Tensor, 
        ric_s: Ricci_Scalar, coords: list[Coordinate]
    ) -> MDMA:
        n, g, Ric, R = len(coords), metric.use, ric_t.use, ric_s.value
        G = MDMA.zeros(n, n)
        for i in product(range(n), repeat=2):
            G[i] = simplify(Ric[i] - S(1)/2 * R * g[i])
        return G


class Run_Calc:
    def __init__(
        self, coords: list, g_m: Matrix, alt_basis: Basis = None
    ) -> None:
        self.n = len(coords)
        self.coords = coords
        self.alt_basis = alt_basis
        self.using_alt_basis = alt_basis is not None
        self.metric = Metric(g_m, coords, alt_basis)
        self.inverse_metric = Inverse_Metric(self.metric, coords, alt_basis)
        self.christoffel = Christoffel(
            coords, self.metric, self.inverse_metric
        )
        self.riemann = Riemann(self.christoffel, coords, alt_basis)
        self.ricci_tensor = Ricci_Tensor(self.riemann, coords, alt_basis)
        self.ricci_scalar = Ricci_Scalar(
            self.ricci_tensor, self.inverse_metric.use, alt_basis
        )
        self.einstein = Einstein(
            self.metric, self.ricci_tensor, 
            self.ricci_scalar, coords, alt_basis
        )

    def return_all_GR_tensors(self) -> dict[str, list[str]]:
        output = {}
    
        def safe_add(label: str, data: list[str]) -> None:
            if data:
                output[label] = data
    
        safe_add("Metric", self.metric.finalize_tensor())
        safe_add("Inverse metric", self.inverse_metric.finalize_tensor())
        safe_add("∂ Metric", self.metric.partial_derivative().finalize_array())
        safe_add("Christoffel symbols", self.christoffel.finalize_array())
        safe_add(
            "∂ Christoffel symbols",
            self.christoffel.partial_derivative().finalize_array()
        )
        safe_add("Riemann curvature tensor", self.riemann.finalize_tensor())
        safe_add("Ricci curvature tensor", self.ricci_tensor.finalize_tensor())
        if self.ricci_scalar.value != 0:
            output["Ricci scalar"] = [self.ricci_scalar.finalized]
        safe_add("Einstein tensor", self.einstein.finalize_tensor())
        mixed_index_G = self.einstein.raise_index(
            0, self.inverse_metric.use, self.alt_basis
        )
        safe_add(
            "Mixed-index Einstein tensor", mixed_index_G.finalize_tensor()
        )
        contravar_G = mixed_index_G.raise_index(
            1, self.inverse_metric.use, self.alt_basis
        )
        safe_add(
            "Contravariant Einstein tensor", contravar_G.finalize_tensor()
        )
        return output

    @classmethod
    def from_demo_1(cls) -> 'Run_Calc':
        coords = [
            Coordinate(0, 't'), Coordinate(1, 'l'), 
            Coordinate(2, 'theta'), Coordinate(3, 'phi')
        ]
        return cls(
            coords = coords,
            g_m = Matrix([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 'r(l)^2', 0],
                [0, 0, 0, 'r(l)^2*sin(theta)^2']
            ]),
            alt_basis = Basis([
                Basis_Vector(0, ['1', '0', '0', '0'], coords),
                Basis_Vector(1, ['0', '1', '0', '0'], coords),
                Basis_Vector(2, ['0', '0', '1/r(l)', '0'], coords),
                Basis_Vector(
                    3, ['0', '0', '0', '1/(r(l)*sin(theta))'], coords
                )
            ])
            
        )

    @classmethod
    def from_demo_2(cls) -> 'Run_Calc':
        coords = [Coordinate(0, 'theta'), Coordinate(1, 'phi')]
        return cls(
            coords = coords,
            g_m = Matrix([[1, 0], [0, 'sin(theta)^2']]),
            alt_basis = Basis([
                Basis_Vector(0, ['1', '0'], coords), 
                Basis_Vector(1, ['0', '1/sin(theta)'], coords)
            ])
        )

    @classmethod
    def from_demo_3(cls) -> 'Run_Calc':
        coords = [
            Coordinate(0, 't'), Coordinate(0, 'r'), 
            Coordinate(0, 'theta'), Coordinate(0, 'phi')
        ]
        return cls(
            coords = coords,
            g_m = Matrix([
                ['-f(t,r)', 0, 0, 0],
                [0, 'h(t,r)', 0, 0],
                [0, 0, 'r^2', 0],
                [0, 0, 0, 'r^2*sin(theta)^2']
            ])
        )

    @classmethod
    def from_demo_4(cls) -> 'Run_Calc':
        coords = [Coordinate(0, 'x'), Coordinate(1, 'y')]
        return cls(
            coords = coords,
            g_m = eye(2),
            alt_basis = Basis([
                Basis_Vector(0, ['cos(x)', 'sin(x)'], coords),
                Basis_Vector(0, ['cos(x+pi/2)', 'sin(x+pi/2)'], coords)
            ])
        )


if __name__ == '__main__':
    Run_Calc = Run_Calc.from_demo_1()
    #Run_Calc = Run_Calc.from_demo_2()
    #Run_Calc = Run_Calc.from_demo_3()
    #Run_Calc = Run_Calc.from_demo_4()
    #Run_Calc = Prep_Input.from_stdin()
    
    print(Run_Calc.return_all_GR_tensors())