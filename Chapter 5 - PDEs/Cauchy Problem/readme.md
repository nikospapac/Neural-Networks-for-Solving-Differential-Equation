To run the files, type in python console:

python "File PathName" (will run the network with default parameters)

To see all the available parameters, type in python console:

python "File PathName" --h

Example command for custom network parameters:

python "C:\MyFiles\PDEs_BVP.py" --in_points 200 --b_points 120 --neurons 15 --extralayers 0 --epochs 50 --function Gaussian
