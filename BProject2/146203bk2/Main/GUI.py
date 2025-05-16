'''import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from . import Run  # Adjust import based on location of Run.py

def start_process(dataset, selection_type, value):
    if not dataset or not selection_type or not value:
        return [""] * 18

    try:
        value = float(value)
        tp = value / 100 if selection_type == 'Training data (%)' else (value - 1) / value if value > 1 else 0
        Acc, Tpr, Tnr = Run.callmain(dataset, tp)

        Acc = Acc if isinstance(Acc, (list, np.ndarray)) and len(Acc) == 6 else [""] * 6
        Tpr = Tpr if isinstance(Tpr, (list, np.ndarray)) and len(Tpr) == 6 else [""] * 6
        Tnr = Tnr if isinstance(Tnr, (list, np.ndarray)) and len(Tnr) == 6 else [""] * 6

        return Acc + Tpr + Tnr

    except:
        return [""] * 18

def plot_graph(*args):
    def safe_float(val): return float(val) if str(val).replace('.', '', 1).isdigit() else 0
    Acc = [safe_float(x) for x in args[:6]]
    Tpr = [safe_float(x) for x in args[6:12]]
    Tnr = [safe_float(x) for x in args[12:]]

    data = np.array([Acc, Tpr, Tnr])
    model_labels = [
        'Adaptive E-Bat+DBN', 'CBF+DBN', 'WOA+BRNN',
        'Hybrid NN', 'SSPO-based DQN', 'Proposed RFQN'
    ]
    bar_width = 0.15
    x_labels = np.arange(3)

    plt.figure(figsize=(10, 5), dpi=120)
    for i in range(6):
        plt.bar(x_labels + i * bar_width, data[:, i], width=bar_width, label=model_labels[i])

    plt.xticks(x_labels + bar_width * 2.5, ['Accuracy', 'TPR', 'TNR'])
    plt.ylabel("Scores")
    plt.title("Comparison of Accuracy, TPR, and TNR")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return plt.gcf()

# UI values
dataset_options = ['Adult', 'Credit_Approval']
selection_options = ['Training data (%)', 'K-value']
models = [
    "Adaptive E-Bat+DBN", "CBF+DBN", "WOA+BRNN",
    "Hybrid NN", "SSPO-based DQN", "Proposed RFQN"
]

with gr.Blocks() as demo:
    gr.Markdown("## Credit Approval Analysis")

    with gr.Row():
        dataset = gr.Dropdown(choices=dataset_options, label="Select_dataset")
        selection_type = gr.Dropdown(choices=selection_options, label="Select")
        value = gr.Textbox(placeholder="Enter value", label="")
        start_btn = gr.Button("START")

    # Column headers
    with gr.Row():
        gr.Markdown("**Model Name**")
        gr.Markdown("**Accuracy**")
        gr.Markdown("**TPR**")
        gr.Markdown("**TNR**")

    acc_boxes, tpr_boxes, tnr_boxes = [], [], []

    # One row per model
    for model in models:
        with gr.Row():
            gr.Markdown(f"{model}")
            acc = gr.Textbox(show_label=False)
            tpr = gr.Textbox(show_label=False)
            tnr = gr.Textbox(show_label=False)
            acc_boxes.append(acc)
            tpr_boxes.append(tpr)
            tnr_boxes.append(tnr)

    # Button row and output plot
    with gr.Row():
        run_graph_btn = gr.Button("Run Graph")
        close_btn = gr.Button("Close")
    
    graph_output = gr.Plot()

    start_btn.click(start_process, inputs=[dataset, selection_type, value],
                    outputs=acc_boxes + tpr_boxes + tnr_boxes)

    run_graph_btn.click(plot_graph, inputs=acc_boxes + tpr_boxes + tnr_boxes,
                        outputs=graph_output)

    demo.launch(share=True)
'''

'''
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from . import Run  # Adjust import based on location of Run.py

def start_process(dataset, selection_type, value):
    if not dataset or not selection_type or not value:
        return [""] * 18

    try:
        value = float(value)
        tp = value / 100 if selection_type == 'Training data (%)' else (value - 1) / value if value > 1 else 0
        Acc, Tpr, Tnr = Run.callmain(dataset, tp)

        Acc = Acc if isinstance(Acc, (list, np.ndarray)) and len(Acc) == 6 else [""] * 6
        Tpr = Tpr if isinstance(Tpr, (list, np.ndarray)) and len(Tpr) == 6 else [""] * 6
        Tnr = Tnr if isinstance(Tnr, (list, np.ndarray)) and len(Tnr) == 6 else [""] * 6

        # IMPORTANT: Must return gr.update(value=value) for textboxes
        return [gr.update(value=str(v)) for v in (Acc + Tpr + Tnr)]

    except:
        return [""] * 18

def plot_graph(*args):
    def safe_float(val): return float(val) if str(val).replace('.', '', 1).isdigit() else 0
    Acc = [safe_float(x) for x in args[:6]]
    Tpr = [safe_float(x) for x in args[6:12]]
    Tnr = [safe_float(x) for x in args[12:]]

    data = np.array([Acc, Tpr, Tnr])
    model_labels = [
        'Adaptive E-Bat+DBN', 'CBF+DBN', 'WOA+BRNN',
        'Hybrid NN', 'SSPO-based DQN', 'Proposed RFQN'
    ]
    bar_width = 0.15
    x_labels = np.arange(3)

    plt.figure(figsize=(10, 5), dpi=120)
    for i in range(6):
        plt.bar(x_labels + i * bar_width, data[:, i], width=bar_width, label=model_labels[i])

    plt.xticks(x_labels + bar_width * 2.5, ['Accuracy', 'TPR', 'TNR'])
    plt.ylabel("Scores")
    plt.title("Comparison of Accuracy, TPR, and TNR")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return plt.gcf()

# UI values
dataset_options = ['Adult', 'Credit_Approval']
selection_options = ['Training data (%)', 'K-value']
models = [
    "Adaptive E-Bat+DBN", "CBF+DBN", "WOA+BRNN",
    "Hybrid NN", "SSPO-based DQN", "Proposed RFQN"
]

with gr.Blocks(theme = "soft", css=".custom-btn {background-color: #6474FF; color: white; font-weight: bold; padding: 10px 20px; border-radius: 8px;}") as demo:
    gr.Markdown("<h1 style='text-align: center;'>Credit Approval Analysis</h1>")

    with gr.Row():
        dataset = gr.Dropdown(choices=dataset_options, label="Select_dataset")
        selection_type = gr.Dropdown(choices=selection_options, label="Select")
        value = gr.Textbox(placeholder="Enter value", label="Enter Value")
        start_btn = gr.Button("START", elem_classes=["custom-btn"])

    gr.Markdown("---")

    acc_boxes, tpr_boxes, tnr_boxes = [], [], []

    # First row: Model names horizontally
    with gr.Row():
        gr.Markdown("**Model Name**")
        for model in models:
            gr.Markdown(f"<div style='text-align: center; font-weight: bold;'>{model}</div>")

    # Second row: Accuracy values horizontally
    with gr.Row():
        gr.Markdown("**Accuracy**")
        for _ in models:
            acc = gr.Textbox(show_label=False)
            acc_boxes.append(acc)

    # Third row: TPR values horizontally
    with gr.Row():
        gr.Markdown("**TPR**")
        for _ in models:
            tpr = gr.Textbox(show_label=False)
            tpr_boxes.append(tpr)

    # Fourth row: TNR values horizontally
    with gr.Row():
        gr.Markdown("**TNR**")
        for _ in models:
            tnr = gr.Textbox(show_label=False)
            tnr_boxes.append(tnr)

    gr.Markdown("---")

    # Button row
    with gr.Row():
        run_graph_btn = gr.Button("Run Graph", elem_classes=["custom-btn"])
        close_btn = gr.Button("Close", elem_classes=["custom-btn"])

    graph_output = gr.Plot()

    # Button connections
    start_btn.click(start_process, inputs=[dataset, selection_type, value],
                    outputs=acc_boxes + tpr_boxes + tnr_boxes)

    run_graph_btn.click(plot_graph, inputs=acc_boxes + tpr_boxes + tnr_boxes,
                        outputs=graph_output)

    demo.launch(share=True)

'''


'''
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from . import Run  # Make sure this import is correct according to your project structure

def start_process(dataset, selection_type, value):
    if not dataset or not selection_type or not value:
        return [gr.update(value="")] * 18

    try:
        value = float(value)
        tp = value / 100 if selection_type == 'Training data (%)' else (value - 1) / value if value > 1 else 0
        Acc, Tpr, Tnr = Run.callmain(dataset, tp)

        Acc = Acc if isinstance(Acc, (list, np.ndarray)) and len(Acc) == 6 else [""] * 6
        Tpr = Tpr if isinstance(Tpr, (list, np.ndarray)) and len(Tpr) == 6 else [""] * 6
        Tnr = Tnr if isinstance(Tnr, (list, np.ndarray)) and len(Tnr) == 6 else [""] * 6

        results = Acc + Tpr + Tnr
        return [gr.update(value=str(v)) for v in results]

    except Exception as e:
        print(f"Error: {e}")
        return [gr.update(value="")] * 18

def plot_graph(*metrics):
    def safe_float(val):
        try:
            return float(val)
        except ValueError:
            return 0

    Acc = [safe_float(x) for x in metrics[:6]]
    Tpr = [safe_float(x) for x in metrics[6:12]]
    Tnr = [safe_float(x) for x in metrics[12:18]]

    data = np.array([Acc, Tpr, Tnr])
    model_labels = [
        "Adaptive E-Bat+DBN", "CBF+DBN", "WOA+BRNN",
        "Hybrid NN", "SSPO-based DQN", "Proposed RFQN"
    ]
    bar_width = 0.15
    x = np.arange(3)  # Accuracy, TPR, TNR

    plt.figure(figsize=(12, 6))
    for idx, model in enumerate(model_labels):
        plt.bar(x + idx * bar_width, data[:, idx], width=bar_width, label=model)

    plt.xticks(x + bar_width * 2.5, ['Accuracy', 'TPR', 'TNR'])
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()

# UI Elements
dataset_options = ['Adult', 'Credit_Approval']
selection_options = ['Training data (%)', 'K-value']
model_names = [
    "Adaptive E-Bat+DBN", "CBF+DBN", "WOA+BRNN",
    "Hybrid NN", "SSPO-based DQN", "Proposed RFQN"
]

with gr.Blocks(theme="soft", css=".custom-btn {background-color: #6474FF; color: white; font-weight: bold; padding: 10px 20px; border-radius: 8px;}") as demo:
    gr.Markdown("<h1 style='text-align: center;'>Credit Approval Model Analysis</h1>")

    with gr.Row():
        dataset = gr.Dropdown(choices=dataset_options, label="Select Dataset")
        selection_type = gr.Dropdown(choices=selection_options, label="Select Type")
        value = gr.Textbox(placeholder="Enter value...", label="Enter Value")
        start_btn = gr.Button("Start", elem_classes=["custom-btn"])

    gr.Markdown("---")

    acc_boxes, tpr_boxes, tnr_boxes = [], [], []

    # Model names
    with gr.Row():
        gr.Markdown("**Model Name**")
        for model in model_names:
            gr.Markdown(f"<div style='text-align: center; font-weight: bold;'>{model}</div>")

    # Accuracy values
    with gr.Row():
        gr.Markdown("**Accuracy**")
        for _ in model_names:
            acc = gr.Textbox(show_label=False)
            acc_boxes.append(acc)

    # TPR values
    with gr.Row():
        gr.Markdown("**TPR**")
        for _ in model_names:
            tpr = gr.Textbox(show_label=False)
            tpr_boxes.append(tpr)

    # TNR values
    with gr.Row():
        gr.Markdown("**TNR**")
        for _ in model_names:
            tnr = gr.Textbox(show_label=False)
            tnr_boxes.append(tnr)

    gr.Markdown("---")

    # Run Graph and Close
    with gr.Row():
        run_graph_btn = gr.Button("Run Graph", elem_classes=["custom-btn"])
        close_btn = gr.Button("Close", elem_classes=["custom-btn"])

    graph_output = gr.Plot()

    # Bindings
    start_btn.click(
        start_process,
        inputs=[dataset, selection_type, value],
        outputs=acc_boxes + tpr_boxes + tnr_boxes
    )

    run_graph_btn.click(
        plot_graph,
        inputs=acc_boxes + tpr_boxes + tnr_boxes,
        outputs=graph_output
    )

    demo.launch(share=True)
'''


import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from . import Run  # Make sure your import is correct

def start_process(dataset, selection_type, value):
    if not dataset or not selection_type or not value:
        return [gr.update(value="0")] * 18

    try:
        value = float(value)
        tp = value / 100 if selection_type == 'Training data (%)' else (value - 1) / value if value > 1 else 0
        Acc, Tpr, Tnr = Run.callmain(dataset, tp)

        Acc = Acc if isinstance(Acc, (list, np.ndarray)) and len(Acc) == 6 else [0] * 6
        Tpr = Tpr if isinstance(Tpr, (list, np.ndarray)) and len(Tpr) == 6 else [0] * 6
        Tnr = Tnr if isinstance(Tnr, (list, np.ndarray)) and len(Tnr) == 6 else [0] * 6

        results = Acc + Tpr + Tnr
        return [gr.update(value=f"{float(v)*100:.5f}%") for v in results]

    except Exception as e:
        print(f"Error: {e}")
        return [gr.update(value=0)] * 18

def plot_graph(*metrics):
    def safe_float(val):
        try:
            # Remove '%' if present, then convert
            if isinstance(val, str) and "%" in val:
                val = val.replace("%", "")
            return float(val)
        except:
            return 0

    Acc = [safe_float(x) for x in metrics[:6]]
    Tpr = [safe_float(x) for x in metrics[6:12]]
    Tnr = [safe_float(x) for x in metrics[12:18]]

    data = np.array([Acc, Tpr, Tnr])
    model_labels = [
        "Adaptive E-Bat+DBN", "CBF+DBN", "WOA+BRNN",
        "Hybrid NN", "SSPO-based DQN", "Proposed RFQN"
    ]
    bar_width = 0.15
    x = np.arange(3)  # Accuracy, TPR, TNR

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab20.colors

    for idx, model in enumerate(model_labels):
        plt.bar(x + idx * bar_width, data[:, idx], width=bar_width, label=model, color=colors[idx % len(colors)])

    plt.xticks(x + bar_width * 2.5, ['Accuracy', 'TPR', 'TNR'])
    plt.ylabel("Score (%)")
    plt.title("Model Performance Comparison", fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()

    
# UI Elements
dataset_options = ['Adult', 'Credit_Approval']
selection_options = ['Training data (%)', 'K-value']
model_names = [
    "Adaptive E-Bat+DBN", "CBF+DBN", "WOA+BRNN",
    "Hybrid NN", "SSPO-based DQN", "Proposed RFQN"
]

with gr.Blocks(theme="soft", css="""

body {
    background: #f9f9ff;
}

.custom-btn {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: bold;
    padding: 12px 24px;
    border-radius: 12px;
}
.gr-button:hover {
    transform: scale(1.05);
    transition: 0.3s;
}
""") as demo:

    gr.Markdown("<h1 style='text-align: center; color: #4F46E5;'> Credit Approval Model Analysis </h1>")
    
    with gr.Tabs():
        with gr.Tab("Input & Start"):
            with gr.Row():
                dataset = gr.Dropdown(choices=dataset_options, label="Select Dataset")
                selection_type = gr.Dropdown(choices=selection_options, label="Select Type")
                value = gr.Textbox(placeholder="Enter Value", label="Value (Training % or K-value)")
            
            gr.Markdown("👉 **Click on Start and go to the 'Results & Graphs' tab to see the results.**")

            start_btn = gr.Button(" Start", elem_classes=["custom-btn"])

        with gr.Tab("Results & Graphs"):
            gr.Markdown("### 📈 Model Evaluation Metrics")
            acc_bars, tpr_bars, tnr_bars = [], [], []

            # Model Names
            with gr.Row():
                gr.Markdown("**Model Name**")
                for model in model_names:
                    gr.Markdown(f"<div style='text-align: center; font-weight: bold;'>{model}</div>")

            # Accuracy
            with gr.Row():
                gr.Markdown("**Accuracy (%)**")
                for _ in model_names:
                    acc = gr.Textbox(show_label=False)
                    acc_bars.append(acc)

            # TPR
            with gr.Row():
                gr.Markdown("**TPR (%)**")
                for _ in model_names:
                    tpr = gr.Textbox(show_label=False)
                    tpr_bars.append(tpr)

            # TNR
            with gr.Row():
                gr.Markdown("**TNR (%)**")
                for _ in model_names:
                    tnr = gr.Textbox(show_label=False)
                    tnr_bars.append(tnr)

            gr.Markdown("---")
            with gr.Row():
                run_graph_btn = gr.Button("Run Graph", elem_classes=["custom-btn"])
                close_btn = gr.Button("Close", elem_classes=["custom-btn"])

            graph_output = gr.Plot()

    # Bind actions
    start_btn.click(
        start_process,
        inputs=[dataset, selection_type, value],
        outputs=acc_bars + tpr_bars + tnr_bars
    )

    run_graph_btn.click(
        plot_graph,
        inputs=acc_bars + tpr_bars + tnr_bars,
        outputs=graph_output
    )

    demo.launch(share=True)


