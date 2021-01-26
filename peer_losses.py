import tensorflow as tf
from tensorflow.keras import backend as K


def L_DMI(y_true, y_pred):
    """L_DMI: An Information-theoretic Noise-robust Loss Function : https://arxiv.org/abs/1909.03388 
		github: https://github.com/Newbeeer/L_DMI
		this loss function requires pretraining before use.
		user may pretrain the modeel upto a satistfactory accuracy then optimize this loss"""
    classes = K.int_shape(y_true)[-1]
    U = tf.matmul(tf.transpose(y_pred), y_true))
	# Add 1e-6*tf.eye(classes) for numerical stability as tensorflow may throw not invertible error for tf.det
	U = U +(1e-6*tf.eye(classes)) 
    det = tf.math.log(tf.dtypes.cast(tf.math.abs(tf.linalg.det(U)), tf.float32) + 1e-4)
    return -1*det
	
def peer_DMI(y_true, y_pred):
	"""Peer Loss Functions: Learning from Noisy Labels without Knowing Noise Rates : https://arxiv.org/abs/1910.03231 
		github: https://github.com/gohsyi/PeerLoss
		Peer-loss function proposed in above paper, in my experience this function also requires a pretraining to be stable"""
    classes = K.int_shape(y_true)[-1]
    U = ((1/tf.cast(tf.keras.backend.shape(y_true)[0], tf.float32))*tf.matmul(tf.transpose(y_pred), y_true))
	# Add 1e-6*tf.eye(classes) for numerical stability as tensorflow may throw not invertible error for tf.det
	U = U +(1e-6*tf.eye(classes)) 
    det = tf.math.log(tf.dtypes.cast(tf.math.abs(tf.linalg.det(U)), tf.float32) + 1e-4)
    def ftrue(): return tf.multiply(det, -1)
    def fneg(): return det
    r = tf.cond(det<0.0, ftrue, fneg)
    return r
