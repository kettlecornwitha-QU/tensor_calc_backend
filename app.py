#______________________________________________________________________________
from flask import Flask, request, jsonify
from flask_cors import CORS
from run_calc_adapter import run_tensor_calculator
from sympy import Symbol, latex

def generate_metric_labels(n, coords, diagonal=True, use_alt=False):
    labels = []
    index = (
        [f'{i}' for i in range(n)] if use_alt
        else [latex(Symbol(coord)) for coord in coords]
    )
    for i in range(n):
        if diagonal:
            labels += [f'g_{{{index[i]}{index[i]}}}']
        else:
            for j in range(i, n):
                labels += [f'g_{{{index[i]}{index[j]}}}']
    return labels

app = Flask(__name__)
app.secret_key = "super_secret_key"
CORS(app)  # Allow frontend to make requests (e.g. from GitHub Pages)

# Temporary in-memory session store
sessions = {}

def current_step(inputs):
    if "n" not in inputs:
        return "ask_n"
    if "coords" not in inputs:
        return "ask_coords"
    if "use_alt_basis" not in inputs:
        return "ask_alt_basis"

    n = inputs["n"]
    if not inputs["use_alt_basis"]:
        if "metric_diag" not in inputs:
            return "ask_diag"
        if "metric" not in inputs:
            return "ask_metric"
        return "ready"

    if "alt_basis" not in inputs:
        return "ask_alt_basis_vectors"
    if "ortho" not in inputs:
        return "ask_orthonormal"
    if inputs["ortho"]:
        if "is_pseudo_riemannian" not in inputs:
            return "ask_manifold_type"
        return "ready"
    if "metric_in_CB" not in inputs:
        return "ask_basis_metric_type"
    if "metric_diag" not in inputs:
        return "ask_diag"
    if "metric" not in inputs:
        return "ask_metric"
    return "ready"

@app.route("/next", methods=["POST"])
def next_step():
    session_id = request.json.get("session_id")
    user_input = request.json.get("input")

    if session_id not in sessions:
        sessions[session_id] = {}

    inputs = sessions[session_id]
    step = current_step(inputs)

    try:
        if step == "ask_n":
            n = int(user_input)
            if n <= 0:
                raise ValueError("n must be a positive integer")
            inputs["n"] = n

        elif step == "ask_coords":
            if (
                not isinstance(user_input, list)
                or len(user_input) != inputs["n"]
            ):
                raise ValueError(f"Expected {inputs['n']} coordinate labels")
            if not all(label.isalpha() for label in user_input):
                raise ValueError(
                    "All coordinate labels must only contain letters"
                )
            inputs["coords"] = user_input

        elif step == "ask_alt_basis":
            inputs["use_alt_basis"] = bool(user_input)

        elif step == "ask_diag":
            inputs["metric_diag"] = bool(user_input)

        elif step == "ask_metric":
            expected_len = (
                inputs["n"]
                if inputs["metric_diag"]
                else inputs["n"] * (inputs["n"] + 1) // 2
            )
            if (
                not isinstance(user_input, list) 
                or len(user_input) != expected_len
            ):
                raise ValueError(f"Expected {expected_len} metric components")
            inputs["metric"] = user_input

        elif step == "ask_alt_basis_vectors":
            expected = inputs["n"] ** 2
            if not isinstance(user_input, list) or len(user_input) != expected:
                raise ValueError(
                    f"Expected {expected} components for "
                    "alternate basis vectors"
                )
            inputs["alt_basis"] = user_input

        elif step == "ask_manifold_type":
            inputs["is_pseudo_riemannian"] = bool(user_input)

        elif step == "ask_orthonormal":
            inputs["ortho"] = bool(user_input)

        elif step == "ask_basis_metric_type":
            inputs["metric_in_CB"] = bool(user_input)

    except ValueError as e:
        return jsonify({"error": str(e), "step": step, "inputs": inputs})

    step = current_step(inputs)

    if step == "ready":
        try:
            result_latex = run_tensor_calculator(inputs)
            
            import json
            print("RESULTS SENT TO FRONTEND:")
            print(json.dumps(result_latex, indent=2))

            return jsonify({"step": "done", "results": result_latex})
        except Exception as e:
            return jsonify({"error": f"Calculation failed: {str(e)}"}), 500

    if step == "ask_metric":
        n = inputs["n"]
        coords = inputs["coords"]
        diag = inputs["metric_diag"]
        use_alt = not inputs.get("metric_in_CB", True)
        labels = generate_metric_labels(n, coords, diag, use_alt)
        return jsonify(
            {"step": step, "inputs": inputs, "metric_labels": labels}
        )

    return jsonify({"step": step, "inputs": inputs})

if __name__ == "__main__":
    app.run(debug=True)
