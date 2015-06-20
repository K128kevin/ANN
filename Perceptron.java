// very rough skeleton of perceptron class
// still don't know how we want to structure the neural net data structure-wise
// some of these functions could be moved into node or layer classes
// some more functions will definitely need to be added
public class Perceptron {

	// not sure how to do this but let's pass activation function for nodes in as a parameter (lambda?)
	public Perceptron(double learningRate, int[] layerSizes) {
		
	}

	// pushes input data through network to output layer
	public void activateLayers() {
		
	}

	// given a set of inputs and targets...
	// set inputs, activate layers, compare outputs to targets, propogate error/adjust weights
	public void train(double[] ins, double[] targets) {
		double[] errs = new double[targets.length];
		
		backPropogateErrors(errs);
	}

	// go backwards from outputs to inputs, through each layer of the neural network
	// calculate what values the nodes SHOULD have been, update weights using weight update rule
	public void backPropogateErrors(double[] outputErrors) {
		
	}

	// set perceptron's inputs to given input values
	public void setInputs(double[] ins) {
		
	}

	// go through ANN weights and save them to a file in either XML or JSON format (still to be decided)
	public void saveWeights() {
		
	}

}
