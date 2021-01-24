from configargparse import ArgumentParser
import sys
import pickle
import numpy as np
from scipy.optimize import leastsq

sys.path.insert(0, 'assets')
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model

def main(args):
    # Load MANO model (here we load the right hand model)
    model_path = 'assets/mano/MANO_RIGHT.pkl'
    model = load_model(model_path, ncomps=15, flat_hand_mean=False)
    model.betas[:] = np.zeros(model.betas.size)

    # [wrist, thumb, index, middle, ring, pinky]
    selVerts = np.array([210, 231, 287, 735, 260, 87, 320, 270, 364, 465, 290, 498, 555, 202, 589, 671])
    vertWeights = np.array([1.0, 2.0, 2.0, 2.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0])
    points, mask = load_vertex_annotations(args.input)
    selVerts = selVerts[mask]
    selPoints = points[mask, :]
    selWeights = vertWeights[mask]

    # Set initial translation to center of mass
    init = np.zeros(model.pose.size + 3)
    init[:3] = np.mean(selPoints, axis=0)
    print("Fitting {} points.".format(selPoints.shape[0]))
    print("Initial guess: {}".format(init))
    pose, _ = leastsq(_compute_pose_error, init, args=(model, selVerts, selPoints, selWeights, args.lambda_reg), maxfev=10000)
    model.trans[:] = pose[:3]
    model.pose[:] = pose[3:]

    store_model_parameters(args.output, model)


def store_model_parameters(out_file, model):
    meta = {
        "mano_verts3d": np.array(model.r).reshape((-1, 3)),
        "mano_joints3d": np.array(model.J_transformed).reshape((-1, 3)),
        "mano_trans": np.array(model.trans).reshape((-1)),
        "mano_pose": np.array(model.pose).reshape((-1)),
        "mano_shape": np.array(model.betas).reshape((-1)),
        "mano_side": "right",
    }
    try:
        with open(out_file, 'wb') as f:
            pickle.dump(meta, f)
    except:
        print("Error: Could not store metadata in file {}".format(out_file))


def store_posed_model(outmesh_path, model):
    try:
        with open(outmesh_path, 'w') as fp:
            for v in model.r:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in model.f + 1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    except:
        print("Error: Could not store posed MANO model in file {}".format(outmesh_path))


def _compute_pose_error(pose, model, vertsIndices, pointLabels, vertWeights=None, lambda_reg=0.01):
    # Set translation and pose parameters
    model.trans[:] = pose[:3]
    model.pose[:] = pose[3:]
    # Compute error of vertices and corresponding points
    verts = model.r[vertsIndices, :]
    err3d = np.array(verts - pointLabels, dtype=np.float)
    err = np.linalg.norm(err3d, axis=1)
    # Weight errors
    if vertWeights is not None:
        err = vertWeights * err
    # Add regularization
    reg = lambda_reg * np.linalg.norm(pose[6:])
    err = np.concatenate([err, [reg]], axis=0).flatten()
    # Append zeros to returned error to make scipy believe that the problem is not underconstrained
    if err.size < pose.size:
        err.resize(pose.size)
    return list(err)


def load_vertex_annotations(path):
    points = np.zeros((16,3), dtype=np.float)
    mask = np.zeros((16), dtype=np.bool)

    with open(path, "r") as f:
        for i in range(16):
            l = f.readline().split(" ")
            if len(l) == 3:
                for j in range(3):
                    points[i, j] = float(l[j])
                mask[i] = True

    return points, mask


if __name__ == '__main__':
    parser = ArgumentParser(description='Fit MANO to annotated vertices')
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the hand vertex annotation file')
    parser.add_argument('-l', '--lambda_reg', type=float, default=0.01, help='Regularization weight for the pose parameters')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the relative hand pose annotation file')

    args = parser.parse_args()
    print("Parameters:")
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print("{}: {}".format(key, val))

    main(args)