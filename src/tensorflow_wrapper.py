import tensorflow as tf
import numpy as np

def getTrainingSession3(trainX, trainY, numEpochs, learningRate, weights, bias):
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
	#weights = tf.Variable(tf.zeros([numFeatures, numLabels]))
	#bias  = tf.Variable(tf.ones([1, numLabels]))


	# Define operations
	init_OP = tf.global_variables_initializer()
	apply_weights_OP = tf.matmul(x, weights, name="apply_weights")
	add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
	activation_OP = tf.nn.softmax(add_bias_OP, name="activation") # softmax regression function

	# Define cost opeartions
	cost_OP = tf.nn.l2_loss(activation_OP-y, name="squared_error_cost") # L2 loss function = mean squared error
	training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP) # gradient descent

	# Run the session
	sess = tf.Session()
	sess.run(init_OP)
	#correct_OP = tf.equal(tf.arg_max(activation_OP, 1), tf.argmax(y, 1)) # Computes element wise comparison between predicted labels and true labels
	#accuracy_OP = tf.reduce_mean(tf.cast(correct_OP, "float"))  # Computes the mean success rate
	#accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)

	for i in range(numEpochs):
		step = sess.run(training_OP, feed_dict={x: trainX, y: trainY}) # run training

	return sess

def getTrainingSession2(trainX, trainY, numEpochs, learningRate, weights, bias):
	numFeatures = trainX.shape[1]
	numLabels = trainY.shape[1]

	learningRate = tf.train.exponential_decay(learning_rate = learningRate,
												global_step = 1,
												decay_steps = trainX.shape[0],
												decay_rate = 0.95,
												staircase = True)

	x = tf.placeholder(tf.float32, [None, numFeatures])
	y_ = tf.placeholder(tf.float32, [None, numLabels])

	y = tf.nn.softmax(tf.matmul(x, weights) + bias)

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	training = tf.train.GradientDescentOptimizer(learningRate).minimize(cross_entropy) # gradient descent

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for i in range(numEpochs):
		sess.run(training, feed_dict={x: trainX, y_: trainY}) # run training

	return sess

def getTrainingSession(trainX, trainY, numEpochs, learningRate, weights, bias, lossFuncName):
	numFeatures = trainX.shape[1]
	numLabels = trainY.shape[1]

	learningRate = tf.train.exponential_decay(learning_rate = learningRate,
												global_step = 1,
												decay_steps = trainX.shape[0],
												decay_rate = 0.95,
												staircase = True)

	x = tf.placeholder(tf.float32, [None, numFeatures]) # Instances tensor
	y_ = tf.placeholder(tf.float32, [None, numLabels]) # True labels tensor

	y = tf.nn.softmax(tf.matmul(x, weights) + bias) # Predicted label
	training = tf.train.GradientDescentOptimizer(learningRate).minimize(loss(lossFuncName, y, y_))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for i in range(numEpochs):
		sess.run(training, feed_dict={x: trainX, y_: trainY})

	return sess

def loss(name, y, y_):
	"""
	name: Name of the loss function
	y: The predicted labels
	y_: The true labels
	"""

	if name == 'cross_entropy':
		return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	elif name == 'l2':
		return tf.nn.l2_loss(y-y_)

def getResultAccuracy(session, testX, testY, weights, bias):
	resultVector = getResultVector(session, testX, testY, weights, bias)
	#print(tf.cast(resultVector, ""))
	return str(session.run(tf.reduce_mean(tf.cast(resultVector, "float"))))
	#print(str(session.run(tf.reduce_mean(tf.cast(resultVector, "float")))))

def getResultAccuracy2(session, testX, testY, weights, bias):
	resultVector = getResultVector2(session, testX, testY, weights, bias)
	return str(session.run(tf.reduce_mean(tf.cast(resultVector, "float"))))

def getResultVector2(session, testX, testY, weights, bias):
	numFeatures = testX.shape[1]
	numLabels = testY.shape[1]

	x = tf.placeholder(tf.float32, [None, numFeatures])
	y = tf.placeholder(tf.float32, [None, numLabels])

	# Define operations
	apply_weights_OP = tf.matmul(x, weights, name="apply_weights")
	add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
	activation_OP = tf.nn.softmax(add_bias_OP, name="activation") # softmax regression function
	correct_OP = tf.equal(tf.arg_max(activation_OP, 1), tf.argmax(y, 1)) # Computes element wise comparison between predicted labels and true labels

	return session.run(correct_OP, feed_dict={x: testX, y:testY})

def getResultVector(session, testX, testY, weights, bias):
	numFeatures = testX.shape[1]
	numLabels = testY.shape[1]

	x = tf.placeholder(tf.float32, [None, numFeatures])
	y_ = tf.placeholder(tf.float32, [None, numLabels])

	# Define operations
	y = tf.nn.softmax(tf.matmul(x, weights) + bias)
	#activation_OP = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y) # softmax regression function
	predicted = tf.equal(tf.arg_max(y_, 1), tf.argmax(y, 1)) # Computes element wise comparison between predicted labels and true labels

	return session.run(predicted, feed_dict={x: testX, y_:testY})

def regression(trainX, trainY, testX, testY, numEpochs, learningRate, weights, bias):
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
	#weights = tf.Variable(tf.zeros([numFeatures, numLabels]))
	#bias  = tf.Variable(tf.ones([1, numLabels]))


	# Define operations
	init_OP = tf.global_variables_initializer()
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

def getInitWeights(trainX, trainY):
	return tf.Variable(tf.zeros([trainX.shape[1], trainY.shape[1]]))

def getInitBias(trainY):
	return tf.Variable(tf.ones([1, trainY.shape[1]]))

def main():
	trainX = np.array([[1,2],[3,4],[5,6]])
	trainY = np.array([[0,0,0,1,0], [1,0,0,0,0], [0,0,0,0,1]])

	testX = np.array([[1,2],[3,4],[5,6]])
	testY = np.array([[0,0,0,1,0], [1,0,0,0,0], [0,0,0,0,1]])

	numEpochs = 27000

	learningRate = 0.0008

	weights = getInitWeights(trainX, trainY)
	bias = getInitBias(trainY)

	#regression(trainX, trainY, testX, testY, numEpochs, learningRate)
	s = getTrainingSession(trainX, trainY, numEpochs, learningRate, weights, bias)
	print(getResultVector(s, testX, testY, weights, bias))
	print(getResultAccuracy(s, testX, testY, weights, bias))
	#print("final accuracy test on test set: %s" %str(s.run(accuracy_OP, feed_dict={x: testX, y:testY})))

if __name__ == '__main__':
	main()
