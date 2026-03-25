"""Balls module.

This module includes the ball and container classes, fundamental for this project.

"""
import numpy as np
from matplotlib.patches import Circle
from math import isclose
class Ball():
    """Class representing a ball in a 2D space.

    Attributes:
        Position
        Velocity
        Radius
        Mass

    """
    def __init__(self, pos = [0., 0.], vel = np.array([1., 0.]), radius = 1., mass = 1.):
        """Initialises the ball with position, velocity, radius, and mass

        Args:
            pos (list or np.ndarray): Position vector [x, y]. Default is [0., 0.].
            vel (list or np.ndarray): Velocity vector [vx, vy]. Default is [1., 0.].
            radius (float): Radius of the ball. Default is 1.
            mass (float): Mass of the ball. Default is 1.

        Raises:
        ValueError: position has to be an array of length 2
        ValueError: velocity has to be an array of length 2
        """
        if len(pos) is not 2:
            raise ValueError("The Position has to be an array with x and y values")
        if len(vel) is not 2:
            raise ValueError("The Velocity has to be an array with x and y values")
        self.__pos = np.array(pos)
        self.__vel = np.array(vel)
        self.__radius = radius
        self.__mass = mass
        self.__patch = Circle(pos, radius, fc='r')
    def pos(self):
        """Returns the position of the ball."""
        return self.__pos
    def vel(self):
        """Returns the velocity of the ball."""
        return self.__vel
    def radius(self):
        """Returns the radius of the ball."""
        return self.__radius
    def mass(self):
        """Returns the mass of the ball."""
        return self.__mass
    def set_vel(self, vel):
        """Sets the velocity of the ball."""
        if len(vel) is not 2:
            raise ValueError("The velocity, has to be an array with x and y components")
        self.__vel = np.array(vel)
    def move(self, dt):
        """Moves the balls position."""
        self.__pos[0] += self.__vel[0] * float(dt)
        self.__pos[1] += self.__vel[1] * float(dt)
        self.__patch.center = self.__pos
    def patch(self):
        """Returns the matplotlib patch of the ball."""
        return self.__patch
    def time_to_collision(self, other):
        """Returns time to collision with another ball/container
        using equation of motion of the balls."""
        rel_pos = other.pos() - self.pos()
        if type(self) is Ball: # Collision with another ball
            rel_vel = other.vel() - self.vel()
            c = np.dot(rel_pos, rel_pos) - (self.radius() + other.radius())**2
        else: # Collision with container
            rel_vel = other.vel()
            c = np.dot(rel_pos, rel_pos) - (self.radius() - other.radius())**2
        a = np.dot(rel_vel, rel_vel)
        b = 2 * np.dot(rel_pos, rel_vel)
        disc = b**2 - 4*a*c
        # Case 1: no real solutions
        if disc < 0 or isclose(a, 0, abs_tol = 1e-8):
            # objects do not collide
            return None
        elif isclose(disc, 0, abs_tol = 1e-8):
            dt = -b/ (2*a)
            ##dt = round(dt,6)
            return float(round(dt, 6)) if dt > 1e-8 else None
        # return smallest positive time = next collision
        else: 
            # solving the quadratic equation
            t1 = (-b - np.sqrt(disc)) / (2*a)
            t2 = (-b + np.sqrt(disc)) / (2*a)

            if isclose(t1, 0, abs_tol = 1e-8):
                t1 = 0
            if isclose(t2, 0, abs_tol = 1e-8):
                t2 = 0

            if t1 > 0 and t2 > 0:
                return float(min(t1, t2))
            elif t1 > 0:
                return float(t1)
            elif t2 > 0:
                return float(t2)
            else:
                # objects do not collide as collision occurs in the past
                return None
        
    def collide(self, other):
        """Collides the ball with another ball."""
        # Calculate the new velocities after collision
        rel_vel = other.vel() - self.vel()
        rel_pos = other.pos() - self.pos()
        #dist = np.sqrt(np.dot(rel_pos, rel_pos))
        dist = np.linalg.norm(rel_pos)
        
        m1 = 2*other.mass() / (self.mass() + other.mass())
        m2 = 2*self.mass() / (self.mass() + other.mass())

        if isclose(dist, 0, abs_tol=1e-8):
            return
        else:
            normal = rel_pos/dist

        self.set_vel(self.vel() + m1 * np.dot(rel_vel, normal) * normal)
        other.set_vel(other.vel() - m2 * np.dot(rel_vel, normal) * normal)

class Container(Ball):
    """
    A subclass of Ball to represent a container.

    Attributes:
    Radius
    Mass
    Velocity
    Position

    """
    def __init__(self, radius = 10., mass = 1e7, dp = 0.):
        """Initialises the container with position [0,0], velocity [0,0], radius, and mass.
        
        Args:
            radius (float): Radius of the container. Default is 10.
            mass (float): Mass of the container. Default is 1e7.
            dp (float): Initial momentum change (for pressure tracking). Default is 0.
            
        """
        super().__init__(pos=[0., 0.], vel=np.array([0., 0.]), radius=radius, mass=mass)
        # Replace red filled circle with blue unfilled for container appearance
        self.__container = Circle(tuple(self.pos()), radius, fc='b', fill=False)
        self.__dp = float(dp)
        self.__dp_v = np.array([0., 0.])
    def patch(self):
        """Returns the matplotlib patch of the container."""
        return self.__container
    def volume(self): #2D so volume is pi*r**2
        """"Returns the volume of the container."""
        return np.pi*self.radius()**2
    def surface_area(self):
        """Returns the surface area of the container."""
        return 2*np.pi*self.radius()
    def dp_tot(self):
        """Returns the total pressure of the container as scalar."""
        return self.__dp
    def dp_v_tot(self):
        """Returns the total pressure of the container as vector."""
        return self.__dp_v
    def collide(self, other):
        """Collides the container with another ball and updates the accumulated momentum to calculate pressure."""
        rel_pos = other.pos() - self.pos()
        rel_vel = other.vel()

        dist = np.linalg.norm(rel_pos)

        if dist < 1e-8:  # Avoids division by zero
            return

        normal = rel_pos/dist
        v_dot_n = np.dot(rel_vel, normal)

        delta_v = -2 * v_dot_n * normal

        other.set_vel(other.vel() + delta_v)
        delta_p = 2 * other.mass() * abs(v_dot_n)
        delta_p_v = 2 * other.mass() * abs(v_dot_n) * normal

        self.__dp += delta_p
        self.__dp_v += delta_p_v    