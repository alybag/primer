#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
using namespace std;

double expliciteuler(int n, double a, double b, double initial, ostream& output)
{
	//local variables
	double h, x, w; //h=dt, x=t, w=x, a=0, b=9

	//initialize values
	h = (b-a)/(double)(n);
	x = a;
	w = initial;

	//Euler's method
	for (int i=0; i<n; i++)
	{
		x = a + i*h; //update value of x to x_i
		output << x << " " << w << endl; //output x_i and w_i to screen
		w = (1-3*h)*w; //update w to w_{i+1}
	}
	output << b << " " << w << endl;
	return w;
}

int main()
{
	//read in params.dat
	string line;
	ifstream infile ("params.dat");
	if (infile.is_open())
	{
		while ( getline (infile,line) )
		{
			cout << line << '\n';
		}
		infile.close();
	}
	else cout << "Unable to open file";

	//output to output.dat
	ofstream myFileStream("output.dat");
	expliciteuler(9, 0., 9., 1., myFileStream);	//change first argument n to match time step
	myFileStream.close();

	return 0;
}
