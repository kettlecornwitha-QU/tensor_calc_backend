#______________________________________________________________________________
from gr_tensor_calculator import Run_Calc
from prep_input_utils import build_coords, build_alt_basis, build_metric_matrix


def run_tensor_calculator(inputs):
    n = inputs["n"]
    coords = build_coords(labels=inputs["coords"])

    alt_basis = None
    if inputs.get("use_alt_basis"):
        alt_basis = build_alt_basis(
            flat_components=inputs["alt_basis"],
            coords=coords,
            n=n
        )

    g_m = build_metric_matrix(
        n=n,
        component_strings=inputs.get("metric"),
        diagonal=inputs.get("metric_diag"),
        use_alt_basis=inputs.get("use_alt_basis", False),
        alt_basis=alt_basis,
        ortho=inputs.get("ortho"),
        metric_in_CB=inputs.get("metric_in_CB"),
        is_pseudo_riemannian=inputs.get("is_pseudo_riemannian")
    )

    calc = Run_Calc(coords, g_m, alt_basis)
    return calc.return_all_GR_tensors()