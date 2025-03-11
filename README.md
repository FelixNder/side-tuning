Running the Demo Script
To run the demo, navigate to the side-tuning/scripts/demo_56.py script. Before executing, make sure to modify the weights on lines 22 and 25. These lines correspond to the base model and the secondary model, respectively. By changing these weights, you can choose the models you want to test. Simply replace the default weights with the appropriate file paths for the models you wish to use.

Weights for ResNet44, ResNet56, and light convolutional models are available in the assets/pytorch directory. You can use these pre-trained weights by specifying the correct path in the script.

Performing Tests Without Side Models or With a Simple Linear Model
If you want to perform tests without using side models, or if you'd like to use a simpler linear model for your tests, you can use the demo_xx_without_side.py script. This script runs the demo without side models, allowing you to test the base model alone.

Alternatively, if you prefer to use a simple linear model, you can use the linear_model.py script, which will load and test a basic linear model configuration.

Distilling a Model
To distill a model, you will need to run the tlkit/models/distillation.py script. This script performs the distillation process, so make sure to check the configurations and parameters inside the file to customize the distillation process according to your needs.