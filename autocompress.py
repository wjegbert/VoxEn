#!/bin/python3

import numpy as np
import tensorflow as tf


sess = tf.InteractiveSession()
# Constants
COLOR_MAX = tf.constant(255, dtype = tf.float32)
ITERATIONS = 100
STRIDE = [1, 4, 2, 2, 1]
STRIDE2 = [1, 4, 1, 1, 1]
VOXEL_SHAPE = [64, 240, 320, 3]

first = 64
second = 32
third = 16
reducedDimX = 60
reducedDimY = 80
reducedDimZ = 1
BATCH = 1

def weights (input_filter_size, out_filter_size):
	return tf.Variable(
		tf.random_normal(
			shape  = [5, 5, 5, input_filter_size, out_filter_size],
			stddev = 0.01,
			dtype  = tf.float32
		)
	)
 
def biases (shape):
	return tf.Variable(
		tf.zeros(shape = shape, dtype = tf.float32)
	)

def clayer (input, weights, biases, stride):

	conv = tf.nn.conv3d(input, weights, strides = stride, padding = 'SAME')
	return conv + biases

def dlayer(input, output, stride):


	deconv = tf.nn.conv3d_transpose(input, weights(output[-1], input.get_shape()[-1].value), output_shape = output, strides = stride)
	output = deconv + biases(output[-1])
	return output

def dense(x, in_features, out_features):
	mat = tf.Variable(tf.random_normal(shape = [in_features, out_features], stddev = .01, dtype=tf.float32))
	x = tf.cast(x, tf.float32)
	mat = tf.cast(mat, tf.float32)
	output = tf.matmul(x, mat) + biases(out_features)
	
	return output

def encoder (x):
	l1 = tf.nn.elu(clayer(x, weights(3, first), biases(first), STRIDE))
	l2 = tf.nn.elu(clayer(l1, weights(first, second),biases(second), STRIDE2))
	l3 = tf.nn.relu(clayer(l2, weights(second, third), biases(third), STRIDE))
	re = tf.reshape(l3, [BATCH, reducedDimX * reducedDimY * reducedDimZ * third])
	
	return re

def decoder (x):
	x = tf.cast(x, tf.float32)
	matrix = (tf.reshape(x, [BATCH, reducedDimZ, reducedDimX, reducedDimY, third]))
	l1 = tf.nn.elu(dlayer(matrix, [BATCH, reducedDimZ * 4, reducedDimX * 2, reducedDimY * 2, second], STRIDE))
	l2 = tf.nn.elu(dlayer(l1, [BATCH, reducedDimZ * 16, reducedDimX * 2, reducedDimY * 2, first], STRIDE2))
	l3 = tf.nn.sigmoid(dlayer(l2, [BATCH, *VOXEL_SHAPE], STRIDE))
	return l3

# Some variables used only in process()
iteration_index = 0

# Process a chunk of video
# Placeholder for feed
feed = tf.placeholder_with_default(
		tf.ones(
			shape = [BATCH, *VOXEL_SHAPE],
			dtype = tf.float32)
			,
			shape = [BATCH, *VOXEL_SHAPE],
			name = "inputVox"
) 


# Create pipeline
# Read in video voxel to tf
dataset = tf.data.Dataset.from_tensor_slices(feed)
batched = dataset.batch(BATCH).repeat()
iterator = batched.make_initializable_iterator()

nextBatch = iterator.get_next()


# Feed into pipeline
latent_vector = encoder(nextBatch)
q_vector = tf.cast(latent_vector, tf.uint8)
decompressed = decoder(q_vector)
loss = tf.reduce_mean(
	tf.square(
			tf.subtract(
			tf.multiply(
				COLOR_MAX,
				nextBatch
			),
			tf.multiply(
				COLOR_MAX,
				decompressed
			)
		)
	)
)

# Optimize error for next batch
optimizer = tf.contrib.optimizer_v2.AdamOptimizer(0.001) 
minimizer = optimizer.minimize(loss)

tf.global_variables_initializer().run()

saver=tf.train.Saver()

def voxelSaver(voxels, iteration, loss_out):
	compressed = latent_vector.eval()
	
	q = q_vector.eval()
	if (iteration_index == 0):
		bat = nextBatch.eval()
		reco = decompressed.eval()
		bat = bat * 255
		reco = reco *255
		bat = bat.astype(np.uint8)
		reco = reco.astype(np.uint8)
	
		# Save the original
		np.save(
			"data/l_%i_original_loss_%s.npy" % (
				iteration, loss_out
				), bat)
		print("Original min %i, max %s, median %d " % (
			np.min(bat), np.max(bat), np.median(bat)
			))
		# save the reconstruction
		np.save(
			"data/l_%i_reconstructed_loss_%s.npy" % (
				iteration, loss_out
				), reco)
		print("Reconstructed min %i, max %s, median %d " % (
			np.min(reco), np.max(reco), np.median(reco)
			))
		print("Compressed min %i, max %s, median %d " % (
			np.min(compressed), np.max(compressed), np.median(compressed)
			))
		print("q min %i, max %s, median %d " % (
			np.min(q), np.max(q), np.median(q)
			))

		print(len(np.unique(compressed)))
		print(len(np.unique(q)))
	else :
		reco = decompressed.eval()
		reco = reco * 255
		reco = reco.astype(np.uint8)
		# save the reconstruction
		np.save(
			"data/l_%i_reconstructed_loss_%s.npy" % (
				iteration, loss_out
				), reco)
		print("Reconstructed min %i, max %s, median %d " % (
			np.min(reco), np.max(reco), np.median(reco)
			))
		np.save(
			"data/compressed.npy", q)
		print("Compressed min %i, max %s, median %d " % (
			np.min(compressed), np.max(compressed), np.median(compressed)
			))
		print("q min %i, max %s, median %d " % (
			np.min(q), np.max(q), np.median(q)
			))

		print(len(np.unique(compressed)))
		print(len(np.unique(q)))
	print("=====Voxels saved!!======")

saver.restore(
	sess, "Trained_Graphs/TrainedLayers1relu.ckpt")
print("Loaded")

def process(voxels):
	global iteration_index
	
	iterator.initializer.run(feed_dict = {feed : voxels})

	minimizer.run(feed_dict={feed : voxels})
	print("Iteration %i" % iteration_index)

	if (iteration_index <= 7000 and (iteration_index % 58 == 0 or iteration_index % 50 == 0)):
		loss_out = loss.eval()
		print("=====loss at %i: %s=======" % (iteration_index, loss_out))
		a = [n.name for n in tf.get_default_graph().as_graph_def().node]
		print(len(a))
		if (iteration_index % 116 == 0):
			voxelSaver(voxels, iteration_index, loss_out)
		if loss_out <=100:
			saver.save(
			sess, "Trained_Graphs/TrainedLayer2relu.ckpt")
			print("Graph saved successfully.")
	elif (7000 < iteration_index <= 10000 and iteration_index % 116 == 0):
		loss_out = loss.eval()
		print("=====loss at %i: %s=======" % (iteration_index, loss_out))
		if (loss_out <= 500):
			voxelSaver(voxels, iteration_index, loss_out)
			
			saver.save(
			sess, "Trained_Graphs/TrainedLayers1relu.ckpt")
			print("Graph saved successfully.")
	elif (10000 < iteration_index <= 50000 and iteration_index % 116 == 0):
		loss_out = loss.eval()
		print("=====loss at %i: %s=======" % (iteration_index, loss_out))
		if (loss_out <= 200):
			voxelSaver(voxels, iteration_index, loss_out)
			
			saver.save(
			sess, "Trained_Graphs/TrainedLayers1.ckpt")
			print("Graph saved successfully.")
	elif (50000 < iteration_index and iteration_index % 1160 == 0):
		loss_out = loss.eval()
		print("=====loss at %i: %s=======" % (iteration_index, loss_out))
		if (loss_out <= 200):
			voxelSaver(voxels, iteration_index, loss_out)
			saver.save(
			sess, "Trained_Graphs/TrainedLayers1.ckpt")
			print("Graph saved successfully.")
	else:
		pass

	iteration_index += 1