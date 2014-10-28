package test.wind.farm.operation.and.maintainance;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;

public class WeibullFittingTest {

	public static void main(String[] args) {
		// read from file
		double[] data = new double[8760];
		int count = 0;
		try {
			String encoding = "GBK";
			File file = new File("E:\\workspace\\WindFarmOM\\windspeed.txt");
			if (file.isFile() && file.exists()) {
				InputStreamReader read = new InputStreamReader(new FileInputStream(file), encoding);
				BufferedReader buffered_reader = new BufferedReader(read);
				String line_text = null;
				while ((line_text = buffered_reader.readLine()) != null) {
					//System.out.println(line_text);
					data[count++] = Double.parseDouble(line_text);
				}
				read.close();
			} else {
				System.out.println("Cannot find file!");
			}
		} catch (Exception e) {
			System.out.println("Error!");
			e.printStackTrace();
		}
				
		// machine learning parameters
		double inteval = 0.12;
		double learning_rate = 0.1;
		WeibullFitting test = new WeibullFitting(inteval, learning_rate);
		
		// generate CDF
		double[] cdf = test.generateCDF(data);
		double[] x = new double[cdf.length];
		
		// sampling
		for (int i = 0; i < x.length; i++) {
			x[i] = i * inteval;
		}

		// learn parameters
		double[] theta = new double[] {8, 2};
		double[] result = test.gradientDescent(x, cdf, theta, 1000);
		test.printArray(result);
	}

}
