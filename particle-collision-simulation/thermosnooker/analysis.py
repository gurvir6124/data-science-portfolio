"""Analysis Module."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from thermosnooker.simulations import SingleBallSimulation, MultiBallSimulation
from thermosnooker.balls import Container, Ball
from thermosnooker.physics import maxwell

def task9():
    """
    Task 9.

    In this function, you should test your animation. To do this, create a container
    and ball as directed in the project brief. Create a SingleBallSimulation object from these
    and try running your animation. Ensure that this function returns the balls final position and
    velocity.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The balls final position and velocity
    """
    c = Container(radius=10.)
    b = Ball(pos=np.array([-5, 0]), vel=np.array([1, 0.]), radius=1., mass=1.)
    sbs = SingleBallSimulation(container=c, ball=b)

    sbs.run(num_collisions=20, animate=True, pause_time=0.5)

    return np.array(b.pos()), np.array(b.vel())

def task10():
    """
    Task 10.

    In this function we shall test your MultiBallSimulation. Create an instance of this class using
    the default values described in the project brief and run the animation for 500 collisions.

    Watch the resulting animation carefully and make sure you aren't seeing errors like balls sticking
    together or escaping the container.
    """

    mbs = MultiBallSimulation()
    mbs.run(num_collisions=500, animate=True, pause_time=0.5)

def task11():
    """
    Task 11.

    In this function we shall be quantitatively checking that the balls aren't escaping or sticking.
    To do this, create the two histograms as directed in the project script. Ensure that these two
    histogram figures are returned.

    Returns:
        tuple[Figure, Firgure]: The histograms (distance from centre, inter-ball spacing).
    """

    mbs = MultiBallSimulation()
    mbs.run(num_collisions=2000, animate=False, pause_time=0.5)
    balls = mbs.balls()
    c_centre = np.array([0., 0.], dtype = float)
    b_dist = []
    for ball in balls:
        dist = np.sqrt(np.dot(ball.pos() - c_centre, ball.pos() - c_centre))
        b_dist.append(dist)
    FIG11_BALLCENTRE = plt.figure()
    plt.hist(b_dist, bins = 8)
    plt.ylabel('Frequency')
    plt.xlabel('Distance Between Each Ball and the Centre of the Container [m]')
    plt.title('Distribution of Distance from Container Centre')

    b_b_dist = []
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            dist = np.linalg.norm(balls[i].pos() - balls[j].pos())
            b_b_dist.append(dist)
    FIG11_BALLINTER = plt.figure()
    plt.hist(b_b_dist, bins = 20)
    plt.ylabel('Frequency')
    plt.xlabel('Distance Between Each Ball [m]')
    plt.title('Inter Ball Distance Distribution')

    plt.show()
    return FIG11_BALLCENTRE, FIG11_BALLINTER


def task12():
    """
    Task 12.

    In this function we shall check that the fundamental quantities of energy and momentum are conserved.
    Additionally we shall investigate the pressure evolution of the system. Ensure that the 4 figures
    outlined in the project script are returned.

    Returns:
        tuple[Figure, Figure, Figure, Figure]: matplotlib Figures the KE, momentum_x, momentum_y ratios
                                               as well as pressure evolution.
    """

    mbs = MultiBallSimulation()
    Ek_0 = mbs.kinetic_energy()
    p_x_0 = mbs.momentum()[0]
    p_y_0 = mbs.momentum()[1]

    Ek_ratio = []
    p_x_ratio = []
    p_y_ratio = []
    times = []
    pressures = []

    for i in range(500):
        mbs.next_collision()

        t = mbs.time()
        pres = mbs.pressure()
        Ek = mbs.kinetic_energy()
        p_x = mbs.momentum()[0]
        p_y = mbs.momentum()[1]

        times.append(t)
        pressures.append(pres)
        Ek_ratio.append(Ek / Ek_0)
        p_x_ratio.append(p_x / p_x_0)
        p_y_ratio.append(p_y / p_y_0)

    FIG12_PT = plt.figure()
    plt.plot(times, pressures)
    plt.ylabel('Pressure [Pa]')
    plt.xlabel('Time [s]')
    plt.title('Pressure Over Time')

    FIG12_EKT = plt.figure()
    plt.plot(times, Ek_ratio)
    plt.ylabel('Kinetic Energy Ratio')
    plt.xlabel('Time [s]')
    plt.title('Kinetic Energy Ratio Over Time')

    FIG12_PXT = plt.figure()
    plt.ticklabel_format(style='plain', axis='y')
    plt.ylim(0.98, 1.02)
    plt.plot(times, p_x_ratio)
    plt.ylabel('Momentum Ratio along x axis')
    plt.xlabel('Time [s]')
    plt.title('X-Axis Momentum Ratio Over Time')

    FIG12_PYT = plt.figure()
    plt.ticklabel_format(style='plain', axis='y')
    plt.ylim(0.98, 1.02)
    plt.plot(times, p_y_ratio)
    plt.ylabel('Momentum ratio along y axis')
    plt.xlabel('Time [s]')
    plt.title('Y-Axis Momentum Ratio Over Time')
    plt.show()
    return FIG12_PT, FIG12_EKT, FIG12_PXT, FIG12_PYT

def task13():
    """
    Task 13.

    In this function we investigate how well our simulation reproduces the distributions of the IGL.
    Create the 3 figures directed by the project script, namely:
    1) PT plot
    2) PV plot
    3) PN plot
    Ensure that this function returns the three matplotlib figures.

    Returns:
        tuple[Figure, Figure, Figure]: The 3 requested figures: (PT, PV, PN)
    """
    radii = [0.1, 0.5, 1.0]
    speeds = np.linspace(0.1, 300, 15)
    FIG13_PT = plt.figure()
    for radius in radii:
        pressures1 = []
        temp1 = []
        for speed in speeds:
            mbs = MultiBallSimulation(b_radius = float(radius), b_speed = float(speed))
            mbs.run(500, False)
            pressures1.append(mbs.pressure())
            temp1.append(mbs.t_equipartition())
        temp1_i = np.linspace(min(temp1), max(temp1), 100)
        pressures1_i = len(mbs.balls()) * 1.380649e-23 * temp1_i / mbs.container().volume()
        plt.plot(temp1, pressures1, linestyle = 'None', marker='o', label=f"Simulation (r={radius}m)")
        plt.xlabel("Temperature (K)", fontsize=13)
        plt.ylabel("Pressure (Pa)", fontsize=13)
        plt.title('Temperature vs Pressure', fontsize=13)
    plt.plot(temp1_i, pressures1_i, 'k-', label="IGL")
    plt.legend()
    plt.grid(True)

    FIG12_PV = plt.figure()
    for radius in radii:
        c_radii = np.linspace(10, 20, 15)
        pressures2 = []
        vol2 = []
        for c_radius in c_radii:
            mbs = MultiBallSimulation(c_radius = float(c_radius), b_radius = float(radius))
            mbs.run(500, False)
            pressures2.append(mbs.pressure())
            V = np.pi * c_radius ** 2
            vol2.append(V)
        vol2_i = np.linspace(min(vol2), max(vol2), 100)
        pressures2_i = len(mbs.balls()) * 1.380649e-23 * mbs.t_equipartition() / np.array(vol2_i)
        plt.plot(vol2, pressures2, linestyle = 'None', marker='o', label=f"Simulation (r={radius}m)")
        plt.xlabel("Volume (m^3)", fontsize=13)
        plt.ylabel("Pressure (Pa)", fontsize=13)
        plt.title('Volume vs Pressure', fontsize=13)
    plt.plot(vol2_i, pressures2_i, 'k-', label="IGL")
    plt.legend()
    plt.grid(True)

    FIG13_PN = plt.figure()
    for radius in radii:
        diff_multi = list(range(1, 9))
        pressures3 = []
        no_balls = []
        for multi in diff_multi:
            mbs = MultiBallSimulation(multi = int(multi), b_radius = float(radius))
            mbs.run(500, False)
            pressures3.append(mbs.pressure())
            N = len(mbs.balls())
            no_balls.append(N)
        no_balls_i = np.linspace(min(no_balls), max(no_balls), 100)
        pressures3_i = np.array(no_balls_i) * 1.380649e-23 * mbs.t_equipartition() / mbs.container().volume()
        plt.plot(no_balls, pressures3, linestyle = 'None', marker = 'o', label=f"Simulation (r={radius}m)")
        plt.xlabel("No. of Balls", fontsize=13)
        plt.ylabel("Pressure (Pa)", fontsize=13)
        plt.title('Number of Balls vs Pressure', fontsize=13)
    plt.plot(no_balls_i, pressures3_i, 'k-',  label="IGL")
    plt.legend()
    plt.grid(True)
    plt.show()

    return FIG13_PT, FIG12_PV, FIG13_PN

def task14():
    """
    Task 14.

    In this function we shall be looking at the divergence of our simulation from the IGL. We shall
    quantify the ball radii dependence of this divergence by plotting the temperature ratio defined in
    the project brief.

    Returns:
        Figure: The temperature ratio figure.
    """
    radii = np.linspace(0.01, 1.0, 15)
    FIG14_T = plt.figure()
    temp_ratio = []
    valid_radii = []
    for radius in radii:
        mbs = MultiBallSimulation(b_radius = float(radius))
        mbs.run(500, False)
        if mbs.t_ideal() == 0:
            continue
        t_ratio = mbs.t_equipartition() / mbs.t_ideal()
        valid_radii.append(radius)
        temp_ratio.append(t_ratio)
    plt.plot(valid_radii, temp_ratio, marker = 'o')
    plt.xlabel("Ball Radius (m)", fontsize=13)
    plt.ylabel("T equipartition/ T ideal", fontsize=13)
    plt.legend()
    plt.grid(True)
    plt.title('Divergence of Simulated Temperature from the Ideal Gas Prediction', fontsize=13)
    plt.show()

    return FIG14_T


def task15():
    """
    Task 15.

    In this function we shall plot a histogram to investigate how the speeds of the balls evolve from the initial
    value. We shall then compare this to the Maxwell-Boltzmann distribution. Ensure that this function returns
    the created histogram.

    Returns:
        Figure: The speed histogram.
    """
    FIG15 = plt.figure()
    init_speeds = [10, 20, 30]
    colours = ['r', 'b', 'g']
    line_styles = ['-', '--', ':']
    edge_colours = ['darkred', 'darkblue', 'darkgreen']
    for i, speed in enumerate(init_speeds):
        colour = colours[i]
        line_style = line_styles[i]
        edge_colour = edge_colours[i]
        mbs = MultiBallSimulation(b_speed = speed)
        mbs.run(1000, False)
        speeds = mbs.speeds()
        plt.hist(speeds, bins = 30, density = True, color = colour, alpha=0.3, edgecolor = edge_colour, label=f'Initial speed = {speed} m/s')
        kbt = mbs.t_equipartition() * 1.380649e-23
        speed_range = np.linspace(0, max(speeds) * 1.4, 200)

        mb_dist = np.array([maxwell(s, kbt, 1.0) for s in speed_range])
        plt.plot(speed_range, mb_dist, color = colour, linestyle = line_style, linewidth = 2.5, label = f'MB Theory (T={kbt:.1f} K)')
        plt.xlabel('Speed (m/s)', fontsize=13)
        plt.ylabel('Probability Density', fontsize=13)
        plt.title('Speed Distribution vs Maxwell-Boltzmann', fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    plt.show()
    return FIG15


def task16():
    """
    Task 16.

    In this function we shall also be looking at the divergence of our simulation from the IGL. We shall
    quantify the ball radii dependence of this divergence by plotting the temperature ratio
    and volume fraction defined in the project brief. We shall fit this temperature ratio before
    plotting the VDW b parameters radii dependence.

    Returns:
        tuple[Figure, Figure]: The ratio figure and b parameter figure.
    """
    from scipy.optimize import curve_fit
    radii = np.linspace(0.01, 1.0, 10)
    FIG16_1 = plt.figure()
    temp_ratio = []
    vol_fracs = []
    def model_func(x, a, b, c):
        return a * x ** 2 + b * x + c
    for radius in radii:
        mbs = MultiBallSimulation(b_radius = float(radius))
        mbs.run(500, False)
        t_ratio = mbs.t_equipartition() / mbs.t_ideal() if mbs.t_ideal() > 0 and np.isfinite(mbs.t_ideal()) else 0
        temp_ratio.append(t_ratio)
        V_ball = np.pi * radius**2
        B = 2 * len(mbs.balls()) * V_ball
        vol_frac = (mbs.container().volume() - B) / mbs.container().volume()
        vol_fracs.append(vol_frac)
    plt.plot(radii, temp_ratio, linestyle = 'None', marker = 'o', label='Temp Ratio')
    plt.plot(radii, vol_fracs, linestyle = 'None', marker = 'x', label='VDW Volume Fraction')
    popt, _ = curve_fit(model_func, radii, temp_ratio)
    x_smooth = np.linspace(min(radii), max(radii), 200)
    plt.plot(x_smooth, model_func(x_smooth, *popt), '-', label='Fitted curve')
    plt.xlabel("Ball Radius (m)", fontsize=13)
    plt.ylabel("Temperature Ratio & Volume Fraction vs Ball Radius", fontsize=13)
    plt.legend()
    plt.grid(True)
    plt.title('Divergence of Simulated Temperature from the Ideal Gas Prediction', fontsize=13)

    b_fit_list = []
    b_approx_list = []

    for r in radii:
        temp_ratio_fitted = model_func(r, *popt)
        b_fit1 = mbs.container().volume() * (1 - temp_ratio_fitted)
        b_fit = b_fit1/(len(mbs.balls()) * 6.022e23)
        b_fit_list.append(b_fit)

        b_approx1 = 2 * len(mbs.balls()) * np.pi * r ** 2
        b_approx = b_approx1 / (len(mbs.balls()) * 6.022e23)
        b_approx_list.append(b_approx)

    FIG16_2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(radii, b_fit_list, 'o', label='b from Fit',
             color='tab:blue')
    ax2.plot(radii, b_approx_list, 's',
             label='b Approximate', color='tab:orange')
    ax2.set_xlabel('Ball Radius (m)', fontsize=13)
    ax2.set_ylabel('b Parameter (m³/mol)', fontsize=13)
    ax2.set_title('Variation of Van der Waals b with Ball Radius', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0.05, 1.05)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    plt.show()
    return FIG16_1, FIG16_2

def task17():
    """
    Task 17.

    In this function we shall run a Brownian motion simulation and plot the resulting trajectory of the 'big' ball.
    """
    from thermosnooker.simulations import BrownianSimulation

    brs = BrownianSimulation()
    brs.run(10000, False)

    fig, axes = plt.subplots(figsize=(10, 6))
    c_radius = brs.container().radius()
    axes.set_xlim(-c_radius, c_radius)
    axes.set_ylim(-c_radius, c_radius)

    container = Circle((0, 0), c_radius, fill=False, color='red', linewidth=3)
    axes.add_patch(container)

    for ball in brs.balls():
        if ball is not brs.big_ball():
            ball = Circle(ball.pos(), ball.radius(), fill=True, color='red', alpha=0.6)
            axes.add_patch(ball)

    big_ball = brs.big_ball()
    big_ball_circle = Circle(tuple(big_ball.pos()), big_ball.radius(), fill=False, color='blue', linewidth=3)
    axes.add_patch(big_ball_circle)

    positions = brs.bb_positions()
    if len(positions) > 0:
        x_positions = [pos[0] for pos in positions]
        y_positions = [pos[1] for pos in positions]

        axes.plot(x_positions, y_positions, 'g-', linewidth=2, alpha=0.7, label='Big Ball Path')
        axes.plot(x_positions[0], y_positions[0], 'go', markersize=8, label='Start Position')
        axes.plot(x_positions[-1], y_positions[-1], 'rs', markersize=8, label='End Position')

    axes.set_title('Brownian Motion Simulation', fontsize=13)
    axes.set_xlabel('X Position (m)', fontsize=13)
    axes.set_ylabel('Y Position (m)', fontsize=13)
    axes.grid(True, alpha=0.3)
    axes.legend(loc='upper right')
    plt.show()

    return fig


if __name__ == "__main__":

    # Run task 9 function
    BALL_POS, BALL_VEL = task9()

    # Run task 10 function
    # task10()

    # Run task 11 function
    # FIG11_BALLCENTRE, FIG11_INTERBALL = task11()

    # Run task 12 function
    # FIG12_KE, FIG12_MOMX, FIG12_MOMY, FIG12_PT = task12()

    # Run task 13 function
    # FIG13_PT, FIG13_PV, FIG13_PN = task13()

    # Run task 14 function
    # FIG14 = task14()

    # Run task 15 function
    # FIG15 = task15()

    # Run task 16 function
    # FIG16_RATIO, FIG16_BPARAM = task16()

    # Run task 17 function
    # task17()

    plt.show()
    

   

    
