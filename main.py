import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from matplotlib import colors
import random
import os
from scipy import ndimage


class NDVIPredatorPreySimulation:
    def __init__(self, grid_size=50, num_herbivores=100, num_carnivores=20, ndvi_update_frequency=10):
        # Initialize grid and parameters
        self.grid_size = grid_size
        self.num_herbivores = num_herbivores
        self.num_carnivores = num_carnivores
        self.ndvi_update_frequency = ndvi_update_frequency

        # Create grids
        self.ndvi_grid = np.zeros((grid_size, grid_size))
        self.herbivore_grid = np.zeros((grid_size, grid_size))
        self.carnivore_grid = np.zeros((grid_size, grid_size))

        # Track individual agents
        self.herbivores = []
        self.carnivores = []

        # Parameters
        self.herbivore_energy_gain = 5  # Energy gained from consuming vegetation
        self.herbivore_energy_loss = 1  # Energy lost per step
        self.herbivore_reproduce_threshold = 20  # Energy needed to reproduce
        self.herbivore_initial_energy = 10

        self.carnivore_energy_gain = 20  # Energy gained from consuming herbivores
        self.carnivore_energy_loss = 1  # Energy lost per step
        self.carnivore_reproduce_threshold = 30  # Energy needed to reproduce
        self.carnivore_initial_energy = 15

        self.ndvi_growth_rate = 0.02  # Base vegetation growth rate
        self.ndvi_max = 1.0  # Maximum NDVI value

        # Statistics tracking
        self.herbivore_count_history = []
        self.carnivore_count_history = []
        self.ndvi_mean_history = []

        # Initialize the environment
        self.initialize_environment()

    def initialize_environment(self):
        """Initialize the simulation environment with NDVI data and animals"""
        # Generate initial NDVI data (we'll use a simple pattern for demonstration)
        self.generate_ndvi_data()

        # Place herbivores randomly
        for _ in range(self.num_herbivores):
            x, y = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
            self.herbivores.append({
                'x': x,
                'y': y,
                'energy': self.herbivore_initial_energy
            })
            self.herbivore_grid[x, y] += 1

        # Place carnivores randomly
        for _ in range(self.num_carnivores):
            x, y = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
            self.carnivores.append({
                'x': x,
                'y': y,
                'energy': self.carnivore_initial_energy
            })
            self.carnivore_grid[x, y] += 1

    def generate_ndvi_data(self):
        """Generate NDVI data - in a real application, this would load actual NDVI data"""
        # For demonstration, create a gradient pattern with some random variation
        x, y = np.meshgrid(np.linspace(0, 1, self.grid_size), np.linspace(0, 1, self.grid_size))

        # Create a base pattern (e.g., some areas with higher vegetation)
        base = 0.5 * np.sin(5 * np.pi * x) * np.sin(5 * np.pi * y)

        # Add some random variation to make it more realistic
        noise = 0.2 * np.random.rand(self.grid_size, self.grid_size)

        # Combine and normalize to [0, 1] range
        self.ndvi_grid = np.clip(base + noise, 0, 1)

        # Apply smoothing to make it more natural
        self.ndvi_grid = ndimage.gaussian_filter(self.ndvi_grid, sigma=1.0)

    def update_ndvi(self, step):
        """Update NDVI values based on growth and consumption"""
        # Periodic major update (simulating seasonal changes)
        if step % self.ndvi_update_frequency == 0:
            # Add some randomness to simulate environmental changes
            variation = 0.1 * np.random.rand(self.grid_size, self.grid_size) - 0.05
            self.ndvi_grid = np.clip(self.ndvi_grid + variation, 0, self.ndvi_max)

        # Natural growth (logistic growth model)
        growth = self.ndvi_growth_rate * self.ndvi_grid * (1 - self.ndvi_grid / self.ndvi_max)
        self.ndvi_grid = np.clip(self.ndvi_grid + growth, 0, self.ndvi_max)

    def move_herbivores(self):
        """Move herbivores based on NDVI values"""
        new_herbivore_grid = np.zeros((self.grid_size, self.grid_size))
        new_herbivores = []

        for herbivore in self.herbivores:
            # Lose energy per step
            herbivore['energy'] -= self.herbivore_energy_loss

            # Die if no energy
            if herbivore['energy'] <= 0:
                continue

            x, y = herbivore['x'], herbivore['y']

            # Find neighboring cells (Moore neighborhood)
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = (x + dx) % self.grid_size, (y + dy) % self.grid_size  # Wrap around
                    neighbors.append((nx, ny, self.ndvi_grid[nx, ny]))

            # Sort neighbors by NDVI value (herbivores prefer higher NDVI)
            neighbors.sort(key=lambda n: n[2], reverse=True)

            # Add some randomness to movement (80% choose best, 20% random)
            if random.random() < 0.8:
                # Move to the cell with highest NDVI
                move_to = neighbors[0]
            else:
                # Random movement
                move_to = random.choice(neighbors)

            new_x, new_y = move_to[0], move_to[1]

            # Consume vegetation (gain energy)
            energy_gain = self.ndvi_grid[new_x, new_y] * self.herbivore_energy_gain
            herbivore['energy'] += energy_gain

            # Reduce NDVI value due to consumption
            self.ndvi_grid[new_x, new_y] = max(0, self.ndvi_grid[new_x, new_y] - 0.1)

            # Update position
            herbivore['x'], herbivore['y'] = new_x, new_y
            new_herbivore_grid[new_x, new_y] += 1

            # Add to new list
            new_herbivores.append(herbivore)

            # Reproduction
            if herbivore['energy'] > self.herbivore_reproduce_threshold:
                herbivore['energy'] /= 2  # Split energy

                # Create offspring with same position
                offspring = {
                    'x': new_x,
                    'y': new_y,
                    'energy': herbivore['energy']
                }

                new_herbivores.append(offspring)
                new_herbivore_grid[new_x, new_y] += 1

        self.herbivores = new_herbivores
        self.herbivore_grid = new_herbivore_grid

    def move_carnivores(self):
        """Move carnivores based on herbivore locations"""
        new_carnivore_grid = np.zeros((self.grid_size, self.grid_size))
        new_carnivores = []

        for carnivore in self.carnivores:
            # Lose energy per step
            carnivore['energy'] -= self.carnivore_energy_loss

            # Die if no energy
            if carnivore['energy'] <= 0:
                continue

            x, y = carnivore['x'], carnivore['y']

            # Find neighboring cells
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = (x + dx) % self.grid_size, (y + dy) % self.grid_size  # Wrap around
                    neighbors.append((nx, ny, self.herbivore_grid[nx, ny]))

            # Sort neighbors by herbivore count (carnivores prefer higher herbivore density)
            neighbors.sort(key=lambda n: n[2], reverse=True)

            # Add some randomness to movement (70% choose best, 30% random)
            if random.random() < 0.7:
                # Move to the cell with highest herbivore count
                move_to = neighbors[0]
            else:
                # Random movement
                move_to = random.choice(neighbors)

            new_x, new_y = move_to[0], move_to[1]

            # Hunt herbivores (gain energy)
            herbivores_here = self.herbivore_grid[new_x, new_y]
            if herbivores_here > 0:
                # Consume up to 2 herbivores maximum
                herbivores_eaten = min(2, herbivores_here)
                carnivore['energy'] += herbivores_eaten * self.carnivore_energy_gain

                # Remove eaten herbivores
                self.remove_herbivores(new_x, new_y, herbivores_eaten)

            # Update position
            carnivore['x'], carnivore['y'] = new_x, new_y
            new_carnivore_grid[new_x, new_y] += 1

            # Add to new list
            new_carnivores.append(carnivore)

            # Reproduction
            if carnivore['energy'] > self.carnivore_reproduce_threshold:
                carnivore['energy'] /= 2  # Split energy

                # Create offspring with same position
                offspring = {
                    'x': new_x,
                    'y': new_y,
                    'energy': carnivore['energy']
                }

                new_carnivores.append(offspring)
                new_carnivore_grid[new_x, new_y] += 1

        self.carnivores = new_carnivores
        self.carnivore_grid = new_carnivore_grid

    def remove_herbivores(self, x, y, count):
        """Remove herbivores that have been eaten by carnivores"""
        # Find herbivores at this location
        herbivores_here = [h for h in self.herbivores if h['x'] == x and h['y'] == y]

        # Remove them (up to count)
        removed = 0
        for h in herbivores_here:
            self.herbivores.remove(h)
            removed += 1
            if removed >= count:
                break

        # Update herbivore grid
        self.herbivore_grid[x, y] = max(0, self.herbivore_grid[x, y] - count)

    def update(self, step):
        """Run one step of the simulation"""
        self.update_ndvi(step)
        self.move_herbivores()
        self.move_carnivores()

        # Update statistics
        self.herbivore_count_history.append(len(self.herbivores))
        self.carnivore_count_history.append(len(self.carnivores))
        self.ndvi_mean_history.append(np.mean(self.ndvi_grid))

    def run_simulation(self, steps=100, animate=True, save_animation=False):
        """Run the simulation for a specified number of steps"""
        if animate:
            fig = plt.figure(figsize=(15, 10))

            # Create subplots
            ax1 = plt.subplot2grid((2, 3), (0, 0))  # NDVI
            ax2 = plt.subplot2grid((2, 3), (0, 1))  # Herbivores
            ax3 = plt.subplot2grid((2, 3), (0, 2))  # Carnivores
            ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)  # Population history

            # NDVI colormap
            ndvi_cmap = plt.cm.YlGn

            # Initial plots
            ndvi_plot = ax1.imshow(self.ndvi_grid, cmap=ndvi_cmap, vmin=0, vmax=1)
            ax1.set_title('NDVI')
            plt.colorbar(ndvi_plot, ax=ax1)

            herbivore_plot = ax2.imshow(self.herbivore_grid, cmap='Blues')
            ax2.set_title('Herbivores')

            carnivore_plot = ax3.imshow(self.carnivore_grid, cmap='Reds')
            ax3.set_title('Carnivores')

            line1, = ax4.plot([], [], 'b-', label='Herbivores')
            line2, = ax4.plot([], [], 'r-', label='Carnivores')
            line3, = ax4.plot([], [], 'g-', label='Mean NDVI x100')
            ax4.set_xlim(0, steps)
            ax4.set_ylim(0, self.num_herbivores * 2)  # Adjust as needed
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Count')
            ax4.legend()

            def init():
                ndvi_plot.set_data(self.ndvi_grid)
                herbivore_plot.set_data(self.herbivore_grid)
                carnivore_plot.set_data(self.carnivore_grid)
                line1.set_data([], [])
                line2.set_data([], [])
                line3.set_data([], [])
                return ndvi_plot, herbivore_plot, carnivore_plot, line1, line2, line3

            def animate(i):
                self.update(i)

                ndvi_plot.set_data(self.ndvi_grid)

                # Update herbivore plot and adjust colormap scale
                herbivore_plot.set_data(self.herbivore_grid)
                if np.max(self.herbivore_grid) > 0:
                    herbivore_plot.set_clim(0, max(1, np.max(self.herbivore_grid)))

                # Update carnivore plot and adjust colormap scale
                carnivore_plot.set_data(self.carnivore_grid)
                if np.max(self.carnivore_grid) > 0:
                    carnivore_plot.set_clim(0, max(1, np.max(self.carnivore_grid)))

                # Update population history
                x = range(len(self.herbivore_count_history))
                line1.set_data(x, self.herbivore_count_history)
                line2.set_data(x, self.carnivore_count_history)
                # Scale NDVI to be visible on same plot
                line3.set_data(x, [v * 100 for v in self.ndvi_mean_history])

                # Adjust y axis if needed
                max_pop = max(max(self.herbivore_count_history), max(self.carnivore_count_history) * 5)
                ax4.set_ylim(0, max(max_pop * 1.1, 10))

                return ndvi_plot, herbivore_plot, carnivore_plot, line1, line2, line3

            anim = animation.FuncAnimation(fig, animate, frames=steps, init_func=init,
                                           interval=100, blit=False)

            if save_animation:
                writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                anim.save('ndvi_predator_prey.mp4', writer=writer)

            plt.tight_layout()
            plt.show()
        else:
            # Run without animation
            for i in range(steps):
                self.update(i)
                if i % 10 == 0:
                    print(f"Step {i}: {len(self.herbivores)} herbivores, {len(self.carnivores)} carnivores")

    def plot_results(self):
        """Plot the final simulation results"""
        fig = plt.figure(figsize=(15, 10))

        # Create subplots
        ax1 = plt.subplot2grid((2, 2), (0, 0))  # NDVI
        ax2 = plt.subplot2grid((2, 2), (0, 1))  # Animal density
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)  # Population history

        # Plot NDVI
        ndvi_plot = ax1.imshow(self.ndvi_grid, cmap='YlGn', vmin=0, vmax=1)
        ax1.set_title('Final NDVI Distribution')
        plt.colorbar(ndvi_plot, ax=ax1)

        # Plot animal density
        # Combine herbivores and carnivores into one grid
        combined_grid = np.zeros((self.grid_size, self.grid_size, 3))
        combined_grid[:, :, 1] = self.herbivore_grid / (
            np.max(self.herbivore_grid) if np.max(self.herbivore_grid) > 0 else 1)  # Green for herbivores
        combined_grid[:, :, 0] = self.carnivore_grid / (
            np.max(self.carnivore_grid) if np.max(self.carnivore_grid) > 0 else 1)  # Red for carnivores

        ax2.imshow(combined_grid)
        ax2.set_title('Animal Distribution (Red: Carnivores, Green: Herbivores)')

        # Plot population history
        x = range(len(self.herbivore_count_history))
        ax3.plot(x, self.herbivore_count_history, 'g-', label='Herbivores')
        ax3.plot(x, self.carnivore_count_history, 'r-', label='Carnivores')
        ax3.plot(x, [v * 100 for v in self.ndvi_mean_history], 'b--', label='Mean NDVI x100')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Population')
        ax3.legend()
        ax3.set_title('Population and NDVI History')

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize simulation
    sim = NDVIPredatorPreySimulation(
        grid_size=50,
        num_herbivores=150,
        num_carnivores=50,
        ndvi_update_frequency=40
    )

    # Run simulation
    sim.run_simulation(steps=200, animate=True, save_animation=False)

    # Plot final results
    sim.plot_results()