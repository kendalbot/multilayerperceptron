# multiple perceptron network
"Multi-Layer" (3) perceptron network expanding upon single perceptron implementation.


A single perceptron can only learn simple patterns (Linearly separable - See 1) (like AND or OR).

By combining the outputs of multiple perceptrons you are able to solve more complex problems (XOR).

The first "hidden" perceptron learns: "Output 1 if x1=0 AND x2=1"
The second "hidden" perceptron learns: "Output 1 if x1=1 AND x2=0"
The third, output perceptron learns: "Output 1 if EITHER hidden perceptron outputs 1"










1 - A problem is linearly separable when you can draw a single straight line (or plane in higher dimensions) that completely separates the different classes of data points.
