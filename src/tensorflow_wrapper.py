import tensorflow as tf
import numpy as np

def regression(trainX, trainY, testX, testY, numEpochs, learningRate):
	numFeatures = trainX.shape[1]
	numLabels = trainY.shape[1]

	learningRate = tf.train.exponential_decay(learning_rate = learningRate,
												global_step = 1,
												decay_steps = trainX.shape[0],
												decay_rate = 0.95,
												staircase = True)

	x = tf.placeholder(tf.float32, [None, numFeatures])
	y = tf.placeholder(tf.float32, [None, numLabels])

	# Define weight and bias variables
	weights = tf.Variable(tf.zeros([numFeatures, numLabels]))
	bias  = tf.Variable(tf.ones([1, numLabels]))


	# Define operations
	init_OP = tf.initialize_all_variables()
	apply_weights_OP = tf.matmul(x, weights, name="apply_weights")
	add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
	activation_OP = tf.nn.softmax(add_bias_OP, name="activation") # softmax regression function

	# Define cost opeartions
	cost_OP = tf.nn.l2_loss(activation_OP-y, name="squared_error_cost") # L2 loss function = mean squared error
	training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP) # gradient descent

	# Run the session
	sess = tf.Session()
	sess.run(init_OP)
	correct_OP = tf.equal(tf.arg_max(activation_OP, 1), tf.argmax(y, 1)) # Computes element wise comparison between predicted labels and true labels
	accuracy_OP = tf.reduce_mean(tf.cast(correct_OP, "float"))  # Computes the mean success rate
	#accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)

	epoch_values=[]
	accuracy_values=[]

	for i in range(numEpochs):
		step = sess.run(training_OP, feed_dict={x: trainX, y: trainY}) # run training

		if i % 100 == 0: # Print information for a few iterations
			epoch_values.append(i)

			train_accuracy = sess.run([accuracy_OP],
				feed_dict={x: trainX, y: trainY})

			accuracy_values.append(train_accuracy)

			print(train_accuracy)

	# Testing
	print("final accuracy test on test set: %s" %str(sess.run(accuracy_OP, feed_dict={x: testX, y: testY})))

def main():
	trainX = np.array([[1,2],[3,4],[5,6]])
	trainY = np.array([[0,0,0,1,0], [1,0,0,0,0], [0,0,0,0,1]])

	testX = np.array([[1,2],[3,4],[5,6]])
	testY = np.array([[0,0,0,1,0], [1,0,0,0,0], [0,0,0,0,1]])

	numEpochs = 27000

	learningRate = 0.0008

	regression(trainX, trainY, testX, testY, numEpochs, learningRate)

if __name__ == '__main__':
	main()
