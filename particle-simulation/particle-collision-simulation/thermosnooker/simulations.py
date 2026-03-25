"""Simulations Module.

This module contains simulation classes for single ball, multi ball and Brownian ball scenarios.

"""
import matplotlib.pyplot as plt
import numpy as np
from thermosnooker.balls import Container, Ball

class Simulation:
    """Base class for all simulations.

    """
    def next_collision(self):
        """Method implemented in derived classes to compute and process next collision.

        Raises:
        NotImplementedError: next_collision() needs to be implemented in derived classes

        """
        raise NotImplementedError('next_collision() needs to be implemented in derived classes')
    def setup_figure(self):
        """Method implemented in derived classes to setup matplotlib figure for animation.

        Raises:
            NotImplementedError: setup_figure() needs to be implemented in derived classes

        """
        raise NotImplementedError('setup_figure() needs to be implemented in derived classes')
    def run(self, num_collisions, animate=False, pause_time=0.001):
        """Runs the simulation for a given number of collisions."""
        if animate:
            fig, axes = self.setup_figure()
        for _ in range(num_collisions):
            self.next_collision()
            if animate:
                plt.pause(pause_time)
        if animate:
            plt.show()

class SingleBallSimulation(Simulation):
    """Simulation of a single ball moving in a circular container.
    
    Summary:
    Calculates when the next collision will happen and determines between which objects.
    Moves all balls to this point in time.
    Performes the elastic collision.

    """
    def __init__(self, container, ball):
        """Initialises the simulation with a container and a ball."""
        self.__container = container
        self.__ball = ball
    def container(self):
        """Returns the container in the simulation."""
        return self.__container
    def ball(self):
        """Returns the ball in the simulation."""
        return self.__ball
    def setup_figure(self):
        """Sets up the matplotlib figure and axes for an animation of the simulation."""
        rad = self.container().radius()
        fig = plt.figure()
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.add_artist(self.container().patch())
        ax.add_patch(self.ball().patch())
        return fig, ax
    def next_collision(self):
        """Advances the simulation to the next collision between the ball and the container."""

        container = self.__container
        ball = self.__ball
        col_time = container.time_to_collision(ball)

        ball.move(col_time)
        container.collide(ball)

class MultiBallSimulation(Simulation):
    """Multiple Balls in Circular Container Simulation."""
    def __init__(self, c_radius = float(10), b_radius = float(1), b_speed = float(10), b_mass = float(1), rmax=float(8), nrings=int(3), multi=int(6)):
        """Initialises the simulation with a container and a list of balls.

        Args:
            c_radius (float): Radius of the circular container. Default is 10.
            b_radius (float): Radius of each ball. Default is 1.
            b_speed (float): Initial speed of the balls. Default is 10.
            b_mass (float): Mass of each ball. Default is 1.
            rmax (float): Maximum radial distance from the center at which balls can be placed. Default is 8.
            nrings (int): Number of concentric rings used to position the balls. Default is 3.
            multi (int): Number of balls per ring (angular discretization). Default is 6.

        """
        self.__balls = []
        self.__c_radius = c_radius
        self.__container = Container(radius=c_radius)
        self.__b_speed = b_speed
        self.__time = float(0.)
        self.__pressure = float(0.)
        self.__kinetic_energy = float(0.)

        radii = np.linspace(rmax / nrings, rmax, nrings)
        for i, r in enumerate(radii, start=1):
            n_points = multi * i
            for n in range(n_points):
                theta = 2 * np.pi * n / n_points
                x = r * np.cos(theta)
                y = r * np.sin(theta) 
                pos = [x, y]
                vel_angle = np.random.uniform(0, 2 * np.pi)
                vel = np.array([b_speed * np.cos(vel_angle), b_speed * np.sin(vel_angle)])

                ball = Ball(pos=np.array(pos, dtype=float), vel=vel, radius=b_radius, mass=b_mass)

                self.__balls.append(ball)

    def container(self):
        """Returns the container in the simulation."""
        return self.__container
    def balls(self):
        """Returns the list of balls in the simulation."""
        return self.__balls
    def setup_figure(self):
        """Sets up the matplotlib figure and axes for animation."""
        rad = self.container().radius()
        fig = plt.figure()
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.add_artist(self.container().patch())
        for ball in self.__balls:
            ax.add_patch(ball.patch())
        return fig, ax
    def next_collision(self):
        """Advances the simulation to the next collision between the ball and another object and update state variables including time, kinetic energy, momentum and pressure."""
        min_t = float(np.inf)
        colliding_obj = np.array([])

        for i in range(len(self.__balls)):
            for j in range(i + 1, len(self.__balls)):
                ball1 = self.__balls[i]
                ball2 = self.__balls[j]
                t = ball1.time_to_collision(ball2)
                if t is not None and 1e-8 < t < min_t:
                    min_t = t
                    colliding_obj = (ball1, ball2)

        for ball in self.__balls:
            t = self.__container.time_to_collision(ball)
            if t is not None and 1e-8 < t < min_t:
                min_t = t
                colliding_obj = ["Container", ball]

        self.__time += min_t

        for ball in self.__balls:
            ball.move(min_t)
       
        
        if colliding_obj[0] == "Container":
            ball = colliding_obj[1]
            self.__container.collide(ball)
        else:
            ball1, ball2 = colliding_obj
            ball1.collide(ball2)
            
        return True

    def kinetic_energy(self):
        """Returns the total kinetic energy of the system by adding kinetic energies of the balls."""
        self.__kinetic_energy = 0.
        for ball in self.__balls:
            self.__kinetic_energy += 0.5 * ball.mass() * self.__b_speed ** 2
        return self.__kinetic_energy
    def momentum(self):
        """Returns the total momentum of the system by adding momentum of the container and balls."""
        self.__momentum = np.array([0., 0.])
        for ball in self.__balls:
            self.__momentum += ball.mass() * ball.vel()
        self.__momentum += self.__container.dp_v_tot()
        return self.__momentum
    def time(self):
        """Returns the current time of the simulation."""
        return self.__time
    def pressure(self):
        """Returns the pressure on the container."""
        self.__pressure = 0.
        if self.__time == 0:
            return 0
        self.__pressure = self.__container.dp_tot() / (self.__time * self.__container.surface_area())
        return self.__pressure
    def t_equipartition(self):
        """Returns the temperature of the system using equipartition theorem."""
        self.__kinetic_energy = 0.
        for ball in self.__balls:
            self.__kinetic_energy += 0.5 * ball.mass() * self.__b_speed ** 2
        T = self.__kinetic_energy / (1.380649e-23 * len(self.__balls))
        return T
    def t_ideal(self):
        """Returns the ideal temperature of the system using Ideal Gas formula."""
        pres = self.pressure()
        T_ideal = pres * self.__container.volume() / (len(self.__balls) * 1.380649e-23)
        return T_ideal
    def speeds(self):
        """Returns a list of the speeds of all the balls in the similation."""
        return [float(np.linalg.norm(b.vel())) for b in self.balls()]

  


class BrownianSimulation(MultiBallSimulation):
    """
    Simulates Brownian motion with one large ball centred in the container and multiple smaller surrounding balls.

    """

    def __init__(self, c_radius=10.0, b_radius=1.0, b_speed=10.0, b_mass=1.0,
                 rmax=8.0, nrings=3, multi=6, bb_radius=2.0, bb_mass=10.0):
        """Initialises the simulation with a container and balls."""
        super().__init__(c_radius=c_radius)

        self.__big_ball = Ball(pos=np.array([0.0, 0.0]), vel=np.array([0.0, 0.0]), radius=bb_radius, mass=bb_mass)
        self._MultiBallSimulation__balls = [self.__big_ball]
        self.__bb_pos = [self.__big_ball.pos().copy()]

        min_dist = bb_radius + b_radius + 0.1

        for r, theta in self._gen_ring_pos(rmax, nrings, multi):
            if r < min_dist:
                continue
            x = r * np.cos(theta)
            y = r * np.sin(theta) 
            pos = np.array([x, y])
            vel_angle = np.random.uniform(0, 2 * np.pi)
            vel = np.array([b_speed * np.cos(vel_angle), b_speed * np.sin(vel_angle)])

            small_ball = Ball(pos=pos, vel=vel, radius=b_radius, mass=b_mass)
            self._MultiBallSimulation__balls.append(small_ball)

    def _gen_ring_pos(self, rmax, nrings, multi):
        """Generates polar coordinates for concentric rings around the big ball."""
        yield (0.0, 0.0)
        radii = np.linspace(rmax / nrings, rmax, nrings)
        for i, r in enumerate(radii, start=1):
            n_points = multi * i
            for n in range(n_points):
                theta = 2 * np.pi * n / n_points
                yield (r, theta)

    def bb_positions(self):
        """Returns a copy of the big ball's recorded position history."""
        return self.__bb_pos.copy()

    def big_ball(self):
        """Return the big ball."""
        return self.__big_ball

    def next_collision(self):
        """Extends next collision to record big ball positions after each step."""
        t = super().next_collision()
        if t is not None and 1e-8 < t:
            self.__bb_pos.append(self.__big_ball.pos().copy())
        return t

    def setup_figure(self):
        """Set up the matplotlib figure with the big ball and smaller balls."""
        rad = self.container().radius()
        fig = plt.figure(figsize=(10, 10))
        axes = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        axes.add_artist(self.container().patch())

        for ball in self.balls():
            if ball == self.__big_ball:
                ball.patch().set_color('blue')
            else:
                ball.patch().set_color('red')
            axes.add_patch(ball.patch())

        return fig, axes