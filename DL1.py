#Performing matrix multiplication and finding eigen vectors and eigen values using TensorFlow

import tensorflow as tf
print('Matrix Multiplication Demo')
x = tf.constant([[1, 2, 3],
[4, 5, 6]], dtype=tf.float32)
print ('Matrix X:\n', x)
y = tf.constant([[7,8],
[9,10],
[11, 12]], dtype=tf.float32)
print ("Matrix Y:\n", y)
z = tf.matmul(x,y)
print ("Product (x x Y):\n", z)

# --- Eigenvalues and Elgenvectors —-
A = tf.constant([[1, 2],
[5, 4]], dtype=tf.float32)
print("\nMatrix A:\n", A)
e_vals, e_vecs = tf.linalg.eigh(A)
print("\nEigenvalues:\n", e_vals)
print("\nEigenvectors:\n", e_vecs)
