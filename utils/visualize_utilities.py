import visdom
import numpy as np


def visualize_with_visdom(viz, source_points, deformed_points, target_points, step):
    """
    Visualize 3D points using Visdom in separate plots for source, deformed, and target.
    """
    # Ensure the points are detached from the computation graph and moved to CPU
    source_np = source_points[0].cpu().detach().numpy()
    print(len(source_np))
    deformed_np = deformed_points[0].cpu().detach().numpy()
    target_np = target_points[0].cpu().detach().numpy()

    # Create scatter data for each type of point
    source_data = {
        "X": source_np,
        "Y": np.zeros(len(source_np))  # Label 0 for source
    }

    deformed_data = {
        "X": deformed_np,
        "Y": np.ones(len(deformed_np))  # Label 1 for deformed
    }

    target_data = {
        "X": target_np,
        "Y": np.full(len(target_np), 2)  # Label 2 for target
    }

    # Visualize Source Points
    viz.scatter(
        X=source_data["X"],  # X points (source)
        Y=None,  # Labels for source
        opts={
            # "legend": ["Source"],
            "markersize": 3,
            "xlabel": "X",
            "ylabel": "Y",
            "zlabel": "Z",
            "title": f"Source Points (Step {step})",
            "layoutopts": {
                "plotly": {"scene": {"aspectmode": "data"}}
            }
        },
        name="Source Points",
        win="Source Deformation"  # Separate window for source points
    )

    # Visualize Deformed Points
    viz.scatter(
        X=deformed_data["X"],  # X points (deformed)
        Y=None,  # Labels for deformed
        opts={
            # "legend": ["Deformed"],
            "markersize": 3,
            "xlabel": "X",
            "ylabel": "Y",
            "zlabel": "Z",
            "title": f"Deformed Points (Step {step})",
            "layoutopts": {
                "plotly": {"scene": {"aspectmode": "data"}}
            }
        },
        name="Deformed Points",
        win="Deformed Deformation"  # Separate window for deformed points
    )

    # Visualize Target Points
    viz.scatter(
        X=target_data["X"],  # X points (target)
        Y=None,  # Labels for target
        opts={
            # "legend": ["Target"],
            "markersize": 3,
            "xlabel": "X",
            "ylabel": "Y",
            "zlabel": "Z",
            "title": f"Target Points (Step {step})",
            "layoutopts": {
                "plotly": {"scene": {"aspectmode": "data"}}
            }
        },
        name="Target Points",
        win="Target Deformation"  # Separate window for target points
    )
    

