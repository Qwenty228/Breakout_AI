# Breakout Game with Neuro Evolution of Augmented Topology (NEAT) AI

This project demonstrates the implementation of the classic game Breakout using the Pygame library and a Neuro Evolution of Augmented Topology (NEAT) AI algorithm. 
NEAT is a technique used to evolve artificial neural networks for various tasks, and in this case, it's being used to evolve an AI agent capable of playing Breakout.


## How to Run the Game

1. **Setup Environment:**
   Make sure you have Python 3.x installed on your system. It's recommended to use a virtual environment to avoid package conflicts. You can create a virtual environment using the following command:

   ```bash
   python3 -m venv venv
   ```

   Activate the virtual environment:

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

2. **Install Dependencies:**
   Install the required packages using the following command:

   ```bash
   pip install pygame neat-python
   ```

3. **Run the Game:**
   Navigate to the project directory and run the Breakout game with the NEAT AI:

   ```bash
   python main.py
   ```

## Project Structure
- main.py: Main script that runs the game and NEAT AI algorithm.
- config-feedforward: NEAT configuration file with parameters for the neural network evolution.


## Acknowledgments
This project is inspired by the concept of evolving neural networks to play classic games. 
The NEAT algorithm was introduced by Kenneth O. Stanley and Risto Miikkulainen in their paper "Evolving Neural Networks through Augmenting Topologies" (2002).


