package test.wind.farm.operation.and.maintainance;

import java.lang.Math;
import java.util.Arrays;

import Jama.Matrix;

public class WeibullFitting {
	
	private double inteval;
	private double learning_rate;

	public WeibullFitting(double inteval, double learning_rate) {
		this.inteval = inteval;
		this.learning_rate = learning_rate;
	}
	
	/**
	 * compute cost function
	 * @param x wind speed array
	 * @param y cumulated distribution function from data
	 * @param theta weibull parameters
	 * @return cost
	 */
	public double computeCost(double[] x, double[] y, double[] theta) {
		int M = x.length;
		if (M != y.length) {
			throw new IllegalArgumentException("Illegal arguments!");
		}
		
		double cost = 0;
		for (int i = 0; i < M; i++) {
			cost += (theta[0] + x[i] * theta[1] - y[i]) * 
					(theta[0] + x[i] * theta[1] - y[i]);
		}
		cost = cost / M / 2;
		return cost;
	}
	
	/**
	 * compute gradient for parameters
	 * @param x wind speed array
	 * @param y cumulated distribution function from data
	 * @param theta weibull parameters
	 * @return gradient
	 */
	public double[] computeGradient(double[] x, double[] y, double[] theta) {
		int M = x.length;
		if (M != y.length) {
			throw new IllegalArgumentException("Illegal arguments!");
		}
		int N = theta.length;
		
		Matrix gradient_matrix = new Matrix(new double[][] {{0, 0}});
		Matrix theta_matrix = new Matrix(theta, N);
		double[][] x_temp = new double[M][N];
		for (int i = 0; i < M; i++) {
			x_temp[i][0] = 1;
			for (int j = 1; j < N; j++) {
				x_temp[i][j] = x[i];
			}
		}
		Matrix x_matrix = new Matrix(x_temp);
		Matrix y_matrix = new Matrix(y, M);
		
		gradient_matrix = x_matrix.transpose().times((x_matrix.times(theta_matrix)).minus(y_matrix)).times(-learning_rate / M);
		
		return gradient_matrix.getRowPackedCopy();
	}
	
	/**
	 * Simple gradient descent method
	 * @param x wind speed array
	 * @param y cumulated distribution function from data
	 * @param theta initial weibull parameters
	 * @param iteration iteration number
	 * @return weibull parameters c,k
	 */
	public double[] gradientDescent(double[] x, double[] y, double[] theta, int iteration) {
		int M = x.length;
		if (M != y.length) {
			throw new IllegalArgumentException("Illegal arguments!");
		}
		int N = theta.length;
		
		double[] transformation_x = new double[M - 2];
		double[] transformation_y = new double[M - 2];
		
		for (int i = 1; i <= M - 2; i++) {
			transformation_x[i-1] = Math.log(x[i]);
		}
		for (int i = 1; i <= M - 2; i++) {
			transformation_y[i-1] = Math.log(-Math.log(1 - y[i]));
		}
		
		double[] theta_iter = theta;
		for (int i = 0; i < iteration; i++) {
			double[] gradient = computeGradient(transformation_x, transformation_y, theta_iter);
			for (int j = 0; j < N; j++) {
				theta_iter[j] += gradient[j];
			}
		}
		
		theta_iter[0] = Math.exp(-theta_iter[0] / theta_iter[1]);

		return theta_iter;
	}
	
	/**
	 * helper function for debugging, get weibull fitting parameter 
	 * @return inteval
	 */
	public double getInteval() {
		return inteval;
	}
	
	/**
	 * helper function for debugging, print arrray 
	 * @param array an array
	 */
	public void printArray(double[] array) {
		for (int i = 0; i < array.length; i++) {
			System.out.println(array[i]);
		}
	}
	
	/**
	 * helper function for debugging, print arrray 
	 * @param array an array
	 */
	public void printArray(int[] array) {
		for (int i = 0; i < array.length; i++) {
			System.out.println(array[i]);
		}
	}
	
	/**
	 * find maximum value in an array
	 * @param data array
	 * @return maximum value in the input array
	 */
	private double findMax(double[] data) {
		double maximum = 0;
		for (int i = 0; i < data.length; i++) {
			if (maximum < data[i]) {
				maximum = data[i];
			}
		}
		return maximum;
	}
	
	/**
	 * generate cumulative distribution function from input data
	 * @param data wind speed as input
	 * @return cumulative distribution function, whose indices indicate the ith inteval in CDF 
	 */
	public double[] generateCDF(double[] data) {
		double maximum = findMax(data);
		
		double[] cumulative_distribution_function = new double[(int) (maximum / inteval) + 1];
		int[] histogram = new int[(int) (maximum / inteval) + 1];
		Arrays.fill(cumulative_distribution_function, 0);
		
		for (int i = 0; i < data.length; i++) {
			int index = (int) (data[i] / inteval);
			histogram[index]++;
		}
		
		for (int i = 0; i < histogram.length - 1; i++) {
			histogram[i + 1] = histogram[i] + histogram[i + 1];
		}

		for (int i = 0; i < histogram.length; i++) {
			cumulative_distribution_function[i] = (double) ((histogram[i] + 0.0) / data.length);
		}
		
		return cumulative_distribution_function;
	}

}
