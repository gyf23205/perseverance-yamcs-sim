# Navigation filter and fault residuals

The onboard navigation filter of the Perseverance simulation. For running the rover, fault
injection and dataset generation, see [README.md](README.md).

## Contents

- [Why online, at 30 Hz](#why-online-at-30-hz)
- [Frames and conventions](#frames-and-conventions)
- [Extended Kalman filter: general equations](#extended-kalman-filter-general-equations)
- [State and process model (prediction)](#state-and-process-model-prediction)
- [Measurement models (update)](#measurement-models-update)
- [Monitor-only residuals](#monitor-only-residuals)
- [From residuals to alarms](#from-residuals-to-alarms)
- [Telemetry: `/Rover/estimator`](#telemetry-roverestimator)
- [Fitting the parameters](#fitting-the-parameters)
- [Validation in Isaac Sim](#validation-in-isaac-sim)

The rover runs an onboard **extended Kalman filter (EKF)** at the physics rate (30 Hz). It estimates
the rover's motion from its own sensors and compares every sensor with what the estimate predicts
that sensor should read. Those disagreements, the **residuals**, are what a fault detector
consumes. They are downlinked under `/Rover/estimator` and recorded as **observable** columns.

- **Inputs:** IMU, drive-joint rates, drive-joint torques, steer angles, and the wheel rates the
  drive controller commanded.
- **No ground truth.** The only place the pose is read is the offline calibration dump, where it is
  recorded *beside* the sensor data for fitting and never fed to the filter.

**Terms used in this document:**

| Term | Meaning |
|---|---|
| State $\mathbf x$ | the quantities the filter estimates: position, heading, speeds, yaw rate, sensor biases |
| Prediction / update | the two steps of every filter cycle: move the state forward with a motion model (prediction), then correct it with sensor readings (update) |
| Covariance $P$ | the filter's estimate of its own uncertainty about the state |
| Process noise $Q$ | how much the motion model is allowed to be wrong per step |
| Measurement noise $R$ | the variance of a sensor's noise |
| Gain $K$ | how strongly one measurement corrects each state |
| Innovation $\tilde y$ | a sensor reading minus the reading the filter predicted, before the correction |
| Residual | here: an innovation, or one of the monitor-only residuals defined below |
| Normalized innovation | $\tilde y/\sqrt S$, where $S$ is the predicted variance of $\tilde y$; about standard normal on a healthy rover |
| NIS | normalized innovation squared, $\tilde y^2/S$; its average is about 1 when $Q$ and $R$ are right |
| χ² gate | do not use a reading to update the state when its NIS exceeds 10.828, which a healthy sensor does with probability 0.1% (chi-squared distribution, 1 degree of freedom) |
| Random walk | the model for a quantity with no known dynamics: "no change expected, but it may drift", at a rate set by $q$ |
| Specific force | what an accelerometer reads: acceleration minus gravity, so a parked IMU reads $+g$ upward |
| Euler angles | three rotations about coordinate axes, in a fixed order, that together give an orientation (roll, pitch, yaw are one such set) |
| Quaternion | four numbers that encode a 3D rotation; the IMU reports its orientation this way |
| Rotation matrix $C$ | the 3×3 matrix that converts a vector from the IMU's axes to world axes. Written $C$ here because $R$ is the measurement noise |
| Correlation coefficient | a number from −1 to 1 measuring how closely two signals move together: 1 = always together, 0 = unrelated |
| Consider update | a Kalman update in which some states are deliberately not corrected (their gain is set to zero) |
| Joseph form | the covariance update $P \leftarrow (I-KH)P(I-KH)^\top + R KK^\top$, which stays valid for any gain |
| Monitor-only residual | compared against the filter's estimate, but never used to correct it |
| Window | 1 s of 30 Hz samples, summarized by their mean and their largest value |
| CUSUM | cumulative-sum test: a standard method that raises an alarm when the mean of a signal shifts and stays shifted |
| Rocker-bogie | Perseverance's six-wheel suspension: two pivoting arms per side that keep all wheels on uneven ground |
| Scrub | a wheel sliding sideways across the ground while the rover turns, so it travels a different distance than its rotation implies |
| MAD | median absolute deviation; × 1.4826 it estimates the standard deviation while ignoring outliers |

| File (under `OmniLRS/src/mission_specific/perseverance/estimation/`) | Contents |
|---|---|
| `models.py` | state, process model, measurement models, Jacobians (pure numpy) |
| `ekf.py` | the filter: prediction, gated scalar updates, consider updates, reacquisition |
| `monitor.py` | monitor-only residuals (torque model, command tracking), 1 s windows, CUSUM |
| `nav_filter.py` | adapter: reads the robot each step, runs everything, publishes one snapshot per second |
| `scripts/fit_nav_estimator.py` | fits every parameter below from nominal calibration runs |

Tests (plain `python3`, no Isaac Sim): `test/test_nav_models.py`, `test_nav_ekf.py`,
`test_nav_monitor.py` and `test_nav_filter.py` in the OmniLRS repo.

## Why online, at 30 Hz

The downlink cannot support an offline filter. `motor_encoder` is a joint angle wrapped to one
revolution (1024 counts) and sampled at 1 Hz. At scale 0.35 the wheel radius is
$r = 0.2625 \cdot 0.35 = 0.0919$ m. The wrapped angle only unambiguously resolves a wheel turn of
less than half a revolution per sample:

```math
|\dot\theta| < \frac{\pi}{1\,\text{s}} \quad\Longleftrightarrow\quad |v| < \pi r \approx 0.29\ \text{m/s}
```

The mission scripter drives at 0.15–0.35 m/s, and `goto` at 0.3 m/s, so the downlinked encoder
aliases at normal speeds. The filter therefore runs inside the simulation loop, directly after
`fault_injector.update()`, on the unwrapped joint rates (`Robot.get_wheel_joint_velocities`).

## Frames and conventions

Two coordinate frames are used:
- The **world frame** is fixed to the ground: $x$ and $y$ horizontal, $z$ up.
- The **body frame** is fixed to the rover and moves and turns with it.

The rover model and the code label the body frame's axes differently. The model is the USD file
(Universal Scene Description, the 3D scene format Isaac Sim loads), and its frame is attached to the
*base link*, the rover's main chassis part. The code means the drive controller, which uses Ackermann
steering (each steerable wheel turned by its own angle so that all wheels roll around one common turn
centre), and the navigation filter:

| Body-frame labels used by | $x$ | $y$ | $z$ | Origin |
|---|---|---|---|---|
| the USD model | left | **backward** (toward the rear) | up | the model's origin |
| the drive controller and the filter | left | **forward** | up | centre of the mid axle (the line through the two middle wheels) |

- The rover drives along the model's $-y$, which is why `forward_axis_sign` is $-1$.
- Wheel $i$ sits at $(x_i, y_i)$ in the code's labels. The origin is on the mid axle because the
  middle wheels cannot steer and the controller always puts the turn centre on that line; both middle
  wheels then have $y = 0$.
- Heading $\psi$ is the angle of the rover's forward axis in the world frame, counter-clockwise from
  world $+x$.

Other conventions:
- Yaw rate $\omega > 0$ turns left.
- Steer angle $\delta > 0$ turns the wheel toward the left.
- Pitch $\theta > 0$ is nose up; roll $\phi > 0$ is left side up.

**IMU orientation.** The filter needs to know how the rover is tilted and which way it faces. This
part shows how it computes both from the IMU's orientation.

*Step 1: what the simulator reports.* Isaac Sim's IMU gives its orientation as a quaternion (four
numbers that encode a 3D rotation). `Robot.get_imu_readings()` converts it to three angles with SciPy:

```python
e_z, e_y, e_x = Rotation.from_quat(q).as_euler('zyx', degrees=True)
```

$e_x$, $e_y$, $e_z$ are three rotation angles, in degrees. Together they give the IMU's current
orientation as three turns in the world frame:

1. Start with a copy of the world axes $x$, $y$, $z$ attached to the IMU.
2. Turn it by $e_z$ about world $z$.
3. Then turn it by $e_y$ about world $y$.
4. Then turn it by $e_x$ about world $x$.

After these turns, the copy points exactly along the IMU's current $x$, $y$, $z$ axes. So the three
turns take the world axes to the IMU's axes, and the angles say how the IMU's axes are rotated
relative to the world axes.

An angle is a single number, so it is not written in any frame. The axes each turn is made about
are the fixed **world** axes, not the IMU's own axes: in SciPy, lowercase axis letters such as
`'zyx'` mean turns about fixed axes.

*Step 2: the rotation matrix.* The three turns of Step 1, written as matrices. The turn by an angle
$\alpha$ about world $x$, $y$ and $z$ is:

```math
C_x(\alpha)=\begin{bmatrix}1&0&0\\0&\cos\alpha&-\sin\alpha\\0&\sin\alpha&\cos\alpha\end{bmatrix},\quad
C_y(\alpha)=\begin{bmatrix}\cos\alpha&0&\sin\alpha\\0&1&0\\-\sin\alpha&0&\cos\alpha\end{bmatrix},\quad
C_z(\alpha)=\begin{bmatrix}\cos\alpha&-\sin\alpha&0\\\sin\alpha&\cos\alpha&0\\0&0&1\end{bmatrix}
```

Multiplying them gives one matrix that does all three turns at once. The first turn ($z$) is applied
first, so it sits rightmost:

```math
C = C_x(e_x)\; C_y(e_y)\; C_z(e_z)
```

Applying $C$ to the world axes gives the IMU's axes, the same result as Step 1. Each column of $C$ is
one IMU axis written in world coordinates. Therefore $C$ converts any vector written in IMU axes into
world axes: $\mathbf v_{\text{world}} = C\,\mathbf v_{\text{imu}}$.

$e_x$ and $e_y$ are turns about the fixed world $x$ and $y$ axes, but the filter needs the tilts of the
rover's own axes, so $e_x$ and $e_y$ cannot be used directly.

*Step 3: the rover's pitch, roll and yaw.* The IMU is fixed to the rover, but its axes are
labelled differently from the rover's. The rover's forward, left and up unit vectors, written in IMU
axes, are (`estimator.imu.forward/left`):

```math
\mathbf f = (0,-1,0), \qquad \mathbf l = (1,0,0), \qquad \mathbf u = \mathbf f \times \mathbf l = (0,0,1)
```

By Step 2, $C\mathbf f$ and $C\mathbf l$ are the rover's forward and left axes written in world axes.
From them:
- **pitch** $\theta$: the angle of the forward axis above the world horizontal plane (nose up is positive);
- **roll** $\phi$: the angle of the left axis above the world horizontal plane (left side up is positive);
- **yaw** $\psi_{\text{imu}}$: the direction of the forward axis within the horizontal plane,
  counter-clockwise from world $+x$ (called heading elsewhere in this document).

The world $z$ component of a unit vector is the sine of its angle above the horizontal, so:

```math
\theta = \arcsin\!\big((C\mathbf f)_z\big), \qquad
\phi = \arcsin\!\big((C\mathbf l)_z\big), \qquad
\psi_{\text{imu}} = \operatorname{atan2}\!\big((C\mathbf f)_y,\ (C\mathbf f)_x\big)
```

Over the calibration run (4 nominal episodes, driving in all directions), $\theta$ and $\phi$ have a
correlation coefficient of 1.000 with the true pitch and roll, and $\psi_{\text{imu}}$ is within 0.03°
of the true yaw.

The same vectors split the accelerometer reading $\mathbf a$ and the gyro reading $\boldsymbol\omega_g$
(both in IMU axes) into rover axes:

```math
\begin{aligned}
a_f &= \mathbf a\cdot\mathbf f, \qquad a_l = \mathbf a\cdot\mathbf l && \text{forward / left specific force}\\
\omega_{\text{up}} &= \boldsymbol\omega_g\cdot\mathbf u, \qquad \omega_{\text{left}} = \boldsymbol\omega_g\cdot\mathbf l && \text{rotation rate about the up axis (yaw rate) and about the left axis}
\end{aligned}
```

Note: `Robot.get_imu_readings()` also reports $-e_x$, $-e_y$, $e_z$ under the names roll, pitch and
yaw, and the `imu_orientation` telemetry carries them. Those are not the angles above: over the same
calibration run, $-e_x$ had a correlation coefficient of 0.07 with the true pitch, and $-e_y$ 0.10 with
the true roll.

The accelerometer reads **specific force**: the Isaac IMU is read with `read_gravity=True`, so a
parked rover reads $+g$ upward. On a slope it therefore reads $g\sin\theta$ along its forward axis,
which the process model has to subtract ($g = 1.62$ m/s², `--gravity`).

## Extended Kalman filter: general equations

A Kalman filter keeps a Gaussian belief about the state: a best estimate $\hat{\mathbf x}$ and a
covariance $P$ that says how uncertain that estimate is. The **extended** Kalman filter (EKF) applies
it to nonlinear models by linearizing them around the current estimate with their Jacobians. The
sections below give this filter's specific $f$, $h$, $F$, $H$, $Q$ and $R$.

**Model.** At step $k$, with time step $\Delta t$:

```math
\mathbf x_k = f(\mathbf x_{k-1}, \mathbf u_k) + \mathbf w_k, \quad \mathbf w_k \sim \mathcal N(\mathbf 0, Q)
\qquad\qquad
\mathbf z_k = h(\mathbf x_k) + \mathbf n_k, \quad \mathbf n_k \sim \mathcal N(\mathbf 0, R)
```

| Symbol | Size here | Meaning |
|---|---|---|
| $\mathbf x$, $\hat{\mathbf x}$ | 8×1 | true state, and the filter's estimate of it |
| $P$ | 8×8 | covariance of the estimation error, $P = \operatorname{E}[(\mathbf x - \hat{\mathbf x})(\mathbf x - \hat{\mathbf x})^\top]$; its diagonal holds each state's variance |
| $\mathbf u$ | — | inputs that drive the prediction but are not estimated: accelerometer $a_f, a_l$, pitch rate $\omega_{\text{left}}$, roll $\phi$, pitch $\theta$ |
| $f$ | — | process model: predicts the next state from the current one and the inputs |
| $\mathbf w$, $Q$ | 8×1, 8×8 | process noise and its covariance: how wrong $f$ may be per step |
| $\mathbf z$, $h$ | 1×1 per channel | a sensor reading, and the measurement model that predicts it from the state |
| $\mathbf n$, $R$ | 1×1 per channel | measurement noise and its variance |
| $F = \partial f/\partial\mathbf x$ | 8×8 | process Jacobian, evaluated at the current estimate |
| $H = \partial h/\partial\mathbf x$ | 1×8 per channel | measurement Jacobian, evaluated at the predicted estimate |
| superscript $^-$ | — | "predicted": the value after the prediction step, before the update |

**Prediction.** Move the estimate forward with the model, and grow the uncertainty:

```math
\hat{\mathbf x}^- = f(\hat{\mathbf x}, \mathbf u), \qquad P^- = F P F^\top + Q
```

- $F P F^\top$ carries the existing uncertainty through the motion. For example, an uncertain
  heading becomes an uncertain position after driving.
- $+\,Q$ adds the uncertainty the model itself introduces. So $P$ always grows in this step.

**Update.** Compare a reading with its prediction, and correct the estimate:

```math
\tilde{\mathbf y} = \mathbf z - h(\hat{\mathbf x}^-), \qquad S = H P^- H^\top + R, \qquad K = P^- H^\top S^{-1}
```

```math
\hat{\mathbf x} = \hat{\mathbf x}^- + K\tilde{\mathbf y}, \qquad P = (I - KH)\,P^-
```

- $\tilde{\mathbf y}$ is the innovation, and $S$ its predicted covariance: the state uncertainty seen through $H$,
  plus the sensor noise.
- The gain $K$ weighs the two. If the state is uncertain compared with the sensor ($HP^-H^\top \gg R$),
  $K$ is large and the reading moves the estimate a lot; if the sensor is noisy, $K$ is small.
- $K$ also corrects states the sensor does not see directly, through the off-diagonal entries of
  $P^-$. This is how wheel speeds, which measure $v$, correct the accelerometer bias $b_a$.
- $P$ shrinks in this step: a reading always adds information.

**What this filter does differently from the textbook form:**
- **Scalar, sequential updates.** Each sensor channel updates the state as its own scalar measurement,
  one after another, so $S$ is a number and $S^{-1} = 1/S$. This is equivalent to one vector update when
  the channels' noises are independent, and gives every channel its own innovation and gate.
- **Joseph form** for $P$, instead of $(I - KH)P^-$, so $P$ stays symmetric and positive-definite even
  with a modified gain.
- **Gating:** a reading whose NIS $= \tilde y^2/S$ exceeds 10.828 does not update the state.
- **Consider updates:** the wheels' gain on $\psi$, $\omega$ and $b_g$ is set to zero.

These are described in [Measurement models (update)](#measurement-models-update).

## State and process model (prediction)

```math
\mathbf x = [\,p_x,\ p_y,\ \psi,\ v,\ v_{\text{lat}},\ \omega,\ b_g,\ b_a\,]^\top
```

| State | Meaning |
|---|---|
| $p_x, p_y$ | position in an **episode-local** frame, with the origin where the episode started (no truth needed to initialize) |
| $\psi$ | heading of the forward axis, the same definition as `drive_controller._pose()` |
| $v$ | forward speed along the ground |
| $v_{\text{lat}}$ | sideways sliding speed, positive left; zero on a rover with grip |
| $\omega$ | yaw rate about the body up axis |
| $b_g$, $b_a$ | gyro-$z$ bias and forward accelerometer bias |

Every symbol in $\mathbf x$ is the filter's **estimate**, written $\hat b_a$ etc. where the
distinction matters. In particular, $b_g$ and $b_a$ are not read from the fault injector.

**Sensor biases as estimated states.** The filter assumes the IMU may be biased but never knows the
true bias, or whether a fault was injected. It appends the biases to the state and estimates them
with the motion, which is standard practice in inertial navigation (bias augmentation):
- **Assumed form:** each bias is roughly constant and may drift slowly (a random walk). It enters
  the dynamics in a known way: $b_a$ is subtracted from $a_f$ in the prediction, and the gyro-$z$
  measurement model predicts the reading as $\omega + b_g$.
- **Initial value:** $0$, with the prior uncertainty in $P_0$.
- **How it is learned:** neither bias is measured directly. Wheel odometry also constrains $v$ and
  $\omega$. When the integrated IMU keeps disagreeing with the wheels, the Kalman gain assigns part of
  the disagreement to the bias, through $F_{v,b_a} = -\Delta t$ and the gyro measurement Jacobian.
- **Under an IMU fault:** the injected offset changes the raw readings. The filter does not know
  about it, so $\hat b_g$ and $\hat b_a$ move toward the injected offset. That jump, published as
  `bias.gyro` and `bias.accel`, is one of the IMU fault signatures.
- **Limit:** only gyro-$z$ and forward-accelerometer biases are modeled. The injector also biases
  the other accelerometer and gyro axes and the roll/pitch/yaw attitude. The filter has no state to
  absorb these, so they appear as errors in other residuals (see
  [Validation in Isaac Sim](#validation-in-isaac-sim)). For fault detection this is acceptable: the
  goal is that a fault shows in some residual, not that it is estimated away.

The accelerometer and the gyro rate about the left axis, $\omega_{\text{left}}$, are **inputs**. One step of
$\Delta t$ is:

```math
\begin{aligned}
p_x &\leftarrow p_x + (v\cos\theta\cos\psi - v_{\text{lat}}\sin\psi)\,\Delta t\\
p_y &\leftarrow p_y + (v\cos\theta\sin\psi + v_{\text{lat}}\cos\psi)\,\Delta t\\
\psi &\leftarrow \psi + \frac{\omega_{\text{left}}\sin\phi + \omega\cos\phi}{\cos\theta}\,\Delta t\\
v &\leftarrow v + (a_f - b_a - g\sin\theta + \omega\,v_{\text{lat}})\,\Delta t\\
v_{\text{lat}} &\leftarrow v_{\text{lat}} + (a_l - g\sin\phi\cos\theta - \omega\,v)\,\Delta t\\
\omega,\ b_g,\ b_a &\leftarrow \omega,\ b_g,\ b_a \qquad \text{(random walks)}
\end{aligned}
```

What each term does:
- $\cos\theta$ projects the along-slope speed onto the horizontal.
- The $\psi$ equation is the standard relation between body rotation rates and the rate of change
  of the heading angle. It comes from differentiating the yaw–pitch–roll rotation, and is found in
  any aircraft-kinematics text, usually written $\dot\psi = (q\sin\phi + r\cos\phi)/\cos\theta$, with
  $q$ and $r$ the rates about the right and down axes. With left and up axes the signs are the same.
  On flat ground ($\phi = \theta = 0$) it reduces to $\dot\psi = \omega$. The $\omega_{\text{left}}\sin\phi$
  term matters on rough ground, where the rover rocks in pitch while rolled; without it the heading
  NIS was about 240.
- $\pm\omega v$ are the rotating-frame terms: in a steady left turn the accelerometer reads
  $a_l = \omega v$ without any side slip.

**Why planar states on 3D terrain.** The state is planar, but the terrain still enters through the
measured pitch and roll: $\cos\theta$ projects the speed onto the horizontal, $g\sin\theta$ and
$g\sin\phi\cos\theta$ remove gravity along the slope, and the $\psi$ equation accounts for rocking.
For a rigid body on a tilted surface these terms are exact. A full 3D state (altitude, vertical
speed, roll, pitch, 3-axis biases) would add nothing here:
- The rover stays on the ground, so its vertical speed $\dot z = v\sin\theta$ follows from $v$ and $\theta$.
- Nothing measures altitude, so $z$ would be unobservable and give no residual.
- The simulation supplies $\theta$ and $\phi$ exactly, so they need not be estimated (a real rover
  would need attitude states; see the $Q$ notes below).
- Every residual (wheels, gyro, heading) depends only on $v$, $v_{\text{lat}}$, $\omega$ and $\psi$.

What it misses: all six wheels are assumed to move with one rigid-body motion on one contact plane.
On rough ground the rocker-bogie lets each wheel climb or drop on its own local slope, so its rolling
speed differs from the prediction. That error is absorbed into the fitted $\sigma_{\text{wheel}}$,
which makes the wheel residuals less sensitive to faults on very rough terrain. Position is
horizontal only (no altitude).

The filter propagates the covariance $P \leftarrow F P F^\top + Q$ with the analytic Jacobian
$F = \partial f/\partial \mathbf x$, where $F_{ij} = \partial x_{i,\text{next}}/\partial x_j$. Rows and
columns follow the state order $[\,p_x,\ p_y,\ \psi,\ v,\ v_{\text{lat}},\ \omega,\ b_g,\ b_a\,]$.
$F$ equals the identity $I$ except for 12 off-diagonal entries: every equation has the form
$x_i \leftarrow x_i + (\text{rate})\,\Delta t$, and a state's rate does not depend on the state itself.

```math
F =
\begin{bmatrix}
1 & 0 & F_{p_x,\psi} & F_{p_x,v} & F_{p_x,v_{\text{lat}}} & 0 & 0 & 0\\
0 & 1 & F_{p_y,\psi} & F_{p_y,v} & F_{p_y,v_{\text{lat}}} & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 0 & F_{\psi,\omega} & 0 & 0\\
0 & 0 & 0 & 1 & F_{v,v_{\text{lat}}} & F_{v,\omega} & 0 & F_{v,b_a}\\
0 & 0 & 0 & F_{v_{\text{lat}},v} & 1 & F_{v_{\text{lat}},\omega} & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
```

With the entries written out:

```math
F =
\begin{bmatrix}
1 & 0 & -(v\cos\theta\sin\psi + v_{\text{lat}}\cos\psi)\Delta t & \cos\theta\cos\psi\,\Delta t & -\sin\psi\,\Delta t & 0 & 0 & 0\\
0 & 1 & (v\cos\theta\cos\psi - v_{\text{lat}}\sin\psi)\Delta t & \cos\theta\sin\psi\,\Delta t & \cos\psi\,\Delta t & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 0 & \dfrac{\cos\phi}{\cos\theta}\Delta t & 0 & 0\\
0 & 0 & 0 & 1 & \omega\,\Delta t & v_{\text{lat}}\,\Delta t & 0 & -\Delta t\\
0 & 0 & 0 & -\omega\,\Delta t & 1 & -v\,\Delta t & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
```

- The $b_g$ column is zero off the diagonal: the gyro bias enters no prediction equation, only the
  gyro measurement model ($\omega + b_g$), so it is learned in the update step.
- Roll $\phi$ and pitch $\theta$ are IMU inputs, not states, so they have no columns.
- The $\omega$, $b_g$ and $b_a$ rows are identity rows because these states are random walks.

$Q$ is diagonal:

```math
Q = \operatorname{diag}\!\big(q_p\Delta t,\ q_p\Delta t,\ q_\psi\Delta t,\ (\sigma_a\Delta t)^2,\ (\sigma_a\Delta t)^2,\ q_\omega\Delta t,\ q_{b_g}\Delta t,\ q_{b_a}\Delta t\big)
```

- **Speeds, $(\sigma_a\Delta t)^2$: accelerometer noise.** $\sigma_a$ is the standard deviation of one
  accelerometer sample. A reading $a_f = a_{\text{true}} + \epsilon$, $\epsilon \sim \mathcal N(0, \sigma_a^2)$,
  adds the error $\epsilon\,\Delta t$ to $v$, whose variance is $\Delta t^2\sigma_a^2$. In general,
  input noise enters as $G\,\Sigma_u G^\top$ with $G = \partial f/\partial\mathbf u$; here
  $\partial v/\partial a_f = \partial v_{\text{lat}}/\partial a_l = \Delta t$, because the readings enter
  with unit gain (no scale factor or rotation). It is squared because $\sigma_a$ is a standard
  deviation, while each $q$ below is already a variance per second.
  - $\sigma_a$ is measured, not tuned: the fit takes the standard deviation of the high-frequency part
    of $a_f$. Both axes use the same $\sigma_a$ (one sensor), and their noises are assumed independent.
  - A first version used $\sigma_a^2\Delta t$, treating $\sigma_a^2$ as a density; that overstated
    the speed uncertainty by $1/\Delta t = 30\times$.
- **Attitude noise is not in $Q$.** Pitch $\theta$ and roll $\phi$ also enter the speed equations,
  through gravity: a pitch error $\delta\theta$ changes $\dot v$ by about $g\cos\theta\,\delta\theta$.
  This is left out because, in simulation, the attitude is read from the physics engine, not
  estimated. $\theta$ and $\phi$ correlate 1.000 with the truth, so with an error of about 0.03°,
  $g\,\delta\theta \approx 8\times10^{-4}$ m/s², about 1/60 of $\sigma_a$. Two more reasons:
  - A steady attitude error adds a constant to the forward acceleration, which $\hat b_a$ absorbs.
    $Q$ only describes white (step-to-step independent) noise.
  - Large attitude errors come only from injected IMU faults (up to 20°, i.e.
    $g\sin 20^\circ \approx 0.55$ m/s²). Covering them in $Q$ would hide the fault; left out, they
    show in $\hat b_a$ and, for roll, in the wheel side-slip residuals.

  **A real rover would need it.** A real IMU measures only angular rate and specific force; no sensor
  measures attitude. It must be estimated by integrating the gyro, which drifts without bound, and
  by taking tilt from the gravity direction, which is wrong whenever the rover accelerates or
  vibrates. The combination is typically off by 0.1–1°. At 1°, $g\,\delta\theta \approx 0.028$ m/s²,
  comparable to $\sigma_a$. Ignoring it would make the filter overconfident in $v$ (NIS above 1).
  Because this error changes slowly, a $Q$ term $(g\cos\theta\,\sigma_\theta\Delta t)^2$ is only a
  rough fix; real inertial navigation systems instead add attitude-error and gyro-bias states. The
  same applies to heading: an absolute $\psi$ measurement exists here only because the simulation
  provides it.
- The $q$ values are random-walk densities: "predict no change, but allow drift at rate $q$". All
  share this form, but not the same purpose:
  - **Physical:** $q_\omega$ (yaw-rate changes from steering, which is not modeled), $q_\psi$ (heading
    kinematics missed on rough terrain), $q_{b_g}$ and $q_{b_a}$ (slow bias drift). $q_\omega$ and
    $q_\psi$ are fitted so the gyro and heading NIS average 1; $q_{b_g}$ and $q_{b_a}$ are set by hand.
  - **Numerical:** $q_p$ models no physical drift; it only keeps $P$ well conditioned (next bullet).
- **Position, $q_p = 10^{-6}$ m²/s, is for numerical conditioning only.** Position follows exactly
  from $v$, $v_{\text{lat}}$ and $\psi$, whose uncertainty $FPF^\top$ already carries into it; in this
  Euler step no noise enters $p$ directly, so its analytic $Q$ entry is 0. But position is never
  measured, so it becomes almost fully correlated with $v$ and $\psi$ (correlation $\rho \to \pm 1$),
  and the wheel updates keep shrinking $v$. $P$ then has an eigenvalue near 0, which rounding can push
  negative, and the filter diverges. Since $FPF^\top$ is positive semi-definite,
  $\lambda_{\min}(P^-) \ge \lambda_{\min}(Q)$ (Weyl's inequality; $\lambda_{\min}$ is the smallest
  eigenvalue), so a positive $q_p$ puts a floor of
  $q_p\Delta t \approx 3\times10^{-8}$ m² under every eigenvalue of $P^-$. The added drift is
  negligible (7.7 mm std over 60 s). $q_p$ is set by hand: no channel measures position, so the
  NIS-based fit cannot tune it. The Joseph form protects $P$ the same way in the update step.

**Initialization.** On the first step of an episode (after the landing hold or the episode settle),
$\mathbf x = [0, 0, \psi_{\text{imu}}, 0, 0, 0, 0, 0]$ and

```math
P_0 = \operatorname{diag}(10^{-6},\ 10^{-6},\ (2^\circ)^2,\ 0.05^2,\ 0.02^2,\ 0.05^2,\ 0.02^2,\ 0.05^2)
```

## Measurement models (update)

All updates are **scalar and sequential**, one per sensor channel. Every channel then has its own
innovation and gate. Each step runs in this order:
1. Gyro and heading. These pin $\omega$ and $\psi$ first.
2. The 6 wheel rolling speeds.
3. The 6 wheel side-slip pseudo-measurements.

Every channel uses the same scalar update. Below, $z$, $h$, $H$ and $R$ stand for one channel's reading,
measurement model, Jacobian row and noise variance. Each channel's own symbols carry its name as a
subscript: $z_{\text{wheel},i}$, $z_{\text{sideslip},i}$, $z_{\text{gyro}}$ and $z_{\text{heading}}$, and
likewise for $h$, $H$ and $R$. $i$ is the wheel index, 1 to 6.

```math
\tilde y = z - h(\hat{\mathbf x}^-), \qquad S = H P H^\top + R, \qquad \text{NIS} = \tilde y^2 / S
```

```math
K = P H^\top / S, \qquad \hat{\mathbf x} \leftarrow \hat{\mathbf x} + K\tilde y, \qquad
P \leftarrow (I - K H)\,P\,(I - K H)^\top + R\,K K^\top
```

- $\tilde y$ is the innovation: the reading minus the reading predicted from the current estimate.
- $S$ is the predicted variance of $\tilde y$. It is the sum of two parts: $HPH^\top$, the state
  uncertainty seen through the sensor, and $R$, the sensor noise. Here it is a number, so $S^{-1} = 1/S$.
- NIS is $\tilde y^2/S$. On a healthy rover it averages about 1, and a reading with NIS above the
  gate does not update the state.
- $P$ is the covariance as left by the previous channel's update, since the updates run one after
  another.

The Joseph form keeps $P$ symmetric positive-definite for *any* gain, which the consider update below
relies on. Heading innovations are wrapped to $(-\pi, \pi]$.

**Wheel rolling speed: the Ackermann model inverted.** `AckermannModel` (the drive controller)
turns a body motion into wheel commands. The filter needs the opposite direction. A rigid body
moving with $(v, v_{\text{lat}}, \omega)$ gives the wheel at $(x_i, y_i)$ the ground velocity
(forward, left) $= (v - \omega x_i,\ v_{\text{lat}} + \omega y_i)$. A wheel steered by $\delta_i$
rolls along $(\cos\delta_i, \sin\delta_i)$. Its spin rate sensor measures the spin rate $\dot\theta_i$.
If the wheel does not slip, $r\dot\theta_i$ equals the wheel's ground speed along its rolling
direction. The model predicts this speed by projecting the wheel's ground velocity onto that
direction:

```math
z_{\text{wheel},i} = r\,\dot\theta_i, \qquad
h_{\text{wheel},i}(\mathbf x) = \cos\delta_i\,(v - k\,\omega\,x_i) + \sin\delta_i\,(v_{\text{lat}} + k\,\omega\,y_i)
```

- $\delta_i$ is the measured steer angle divided by `steer_sign`; the mid wheels have $\delta = 0$.
- With $v_{\text{lat}} = 0$ and $k = 1$, this is exactly `AckermannModel._solve_wheel` inverted. The
  unit test checks the round trip $h_{\text{wheel},i} = r\cdot$`speed` for arcs of both signs, straights and point
  turns, including the ±90° steer fold, to $10^{-16}$.
- $k$ = `wheel_yaw_scale` = **1.034**. It is fitted because this rover's wheels scrub in turns: they turn
  as if the rover yawed 3.4% faster than the gyro says.

**Wheel side slip: pseudo-measurement 0.** A rolling wheel does not slide along its axle:

```math
z_{\text{sideslip},i} = 0, \qquad h_{\text{sideslip},i}(\mathbf x) = -\sin\delta_i\,(v - k\,\omega\,x_i) + \cos\delta_i\,(v_{\text{lat}} + k\,\omega\,y_i)
```

For the mid wheels this reduces to $h_{\text{sideslip},i} = v_{\text{lat}}$. A steer angle that does not match the
motion, for example a stuck corner, shows up here on that corner.

**Gyro and heading:**

```math
z_{\text{gyro}} = \omega_{\text{up}} = \boldsymbol\omega_g\cdot\mathbf u,\quad h_{\text{gyro}} = \omega + b_g;
\qquad\qquad
z_{\text{heading}} = \psi_{\text{imu}},\quad h_{\text{heading}} = \psi
```

**Jacobian rows $H$.** Each update needs the Jacobian row $H = \partial h/\partial\mathbf x$, a 1×8 row
with $H_j = \partial h/\partial x_j$. Columns follow the state order
$[\,p_x,\ p_y,\ \psi,\ v,\ v_{\text{lat}},\ \omega,\ b_g,\ b_a\,]$, as in $F$. Every $h$ above is linear in
the state: the steer angle $\delta_i$ and the wheel position $(x_i, y_i)$ are inputs, not states. So $H$
is the list of each state's coefficient in $h$, $h(\mathbf x) = H\mathbf x$ exactly, and linearizing
the update adds no error.

```math
\begin{array}{r|cccccccc}
 & p_x & p_y & \psi & v & v_{\text{lat}} & \omega & b_g & b_a\\ \hline
H_{\text{wheel},i}    & 0 & 0 & 0 & \cos\delta_i  & \sin\delta_i & k\,(y_i\sin\delta_i - x_i\cos\delta_i) & 0 & 0\\
H_{\text{sideslip},i} & 0 & 0 & 0 & -\sin\delta_i & \cos\delta_i & k\,(y_i\cos\delta_i + x_i\sin\delta_i) & 0 & 0\\
H_{\text{gyro}}       & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0\\
H_{\text{heading}}    & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0
\end{array}
```

How each row follows from its $h$:
- **Wheel rolling speed.** Expanding $h_{\text{wheel},i}$ and grouping by state gives
  $\cos\delta_i\,v + \sin\delta_i\,v_{\text{lat}} + k\,(y_i\sin\delta_i - x_i\cos\delta_i)\,\omega$. The
  three coefficients are the $v$, $v_{\text{lat}}$ and $\omega$ entries. For a mid wheel ($\delta = 0$,
  $y = 0$) the row is $[\,\dots,\ 1,\ 0,\ -k\,x_i,\ \dots]$: in a left turn ($\omega > 0$) the left wheel
  ($x_i > 0$) rolls slower than $v$ and the right wheel faster.
- **Wheel side slip.** Expanding $h_{\text{sideslip},i}$ the same way gives
  $-\sin\delta_i\,v + \cos\delta_i\,v_{\text{lat}} + k\,(y_i\cos\delta_i + x_i\sin\delta_i)\,\omega$. For a
  mid wheel the row is $[\,\dots,\ 0,\ 1,\ 0,\ \dots]$, i.e. $h = v_{\text{lat}}$.
- **Gyro.** $h_{\text{gyro}} = \omega + b_g$ has coefficient 1 on $\omega$ and on $b_g$.
- **Heading.** $h_{\text{heading}} = \psi$ has coefficient 1 on $\psi$.

What the zero entries mean:
- **$p_x$, $p_y$ are zero in every row:** no sensor measures position. Position is dead reckoning, and
  its uncertainty only grows (see the $q_p$ note in [State and process model](#state-and-process-model-prediction)).
- **$b_a$ is zero in every row:** no sensor reads the accelerometer bias directly. It is learned only
  through the prediction, where $F_{v,b_a} = -\Delta t$ ties it to $v$, which the wheels measure.
- **$b_g$ appears only in the gyro row,** next to $\omega$. The gyro alone cannot tell them apart;
  the heading row separates them over time, because $\psi$ integrates $\omega$ but not $b_g$.
- **$\psi$ appears only in the heading row:** wheel speeds are body-frame quantities and do not depend
  on which way the rover faces.
- The wheel rows' $\omega$ entry is non-zero, so $\omega$'s uncertainty still enters the wheels' $S$.
  The consider update below only stops the wheels from correcting $\omega$.

**Wheel noise grows in turns.** The scrub is not constant (point turns and arcs differ), so each
wheel's variance includes a yaw-proportional term. The same formula gives $R_{\text{wheel},i}$ and
$R_{\text{sideslip},i}$: $H_i$ is that channel's Jacobian row from the table above, $H_{i,\omega}$ its $\omega$ entry, $\sigma$ is
$\sigma_{\text{wheel}}$ or $\sigma_{\text{sideslip}}$, and $\sigma_k$ = `sigma_yaw_scale` = 0.2:

```math
R_i = \sigma^2 + \big(\sigma_k\,\hat\omega\,H_{i,\omega}\big)^2
```

**The wheel updates do not correct $\psi$, $\omega$ or $b_g$.** The wheel updates are consider
updates. The gain $K = PH^\top/S$ is an 8×1 column with one entry per state; in the wheel updates
its entries for these three states are set to zero ($K_\psi = K_\omega = K_{b_g} = 0$), so the wheels correct only $p_x$, $p_y$, $v$, $v_{\text{lat}}$ and $b_a$. The three states are left to the
gyro and the IMU heading, which measure them more precisely; $b_g$ stays observable because the
heading integrates $\omega$. Without this, every wheel fault leaked into the IMU channels in the fault
runs:
- a sinking or slipping rover pulled $\omega$ off the gyro;
- the gyro-bias estimate drifted 0.05–0.12 rad/s on a healthy gyro;
- the heading residual reached 10–20σ.

Wheel residuals are still judged against the gyro's $\omega$.

**Gating.** A measurement with $\text{NIS} > \chi^2_{1,\,0.999} = 10.828$ does not update the state: $\hat{\mathbf x}$ and $P$ stay as
they were. Its innovation is still computed and published. Each channel type handles such readings
differently.

**Rolling speeds: one common prediction, at most 2 wheels left out of the update.** `_update_group()` processes the
6 wheels in two passes:
1. **Report and gate.** All 6 innovations $\tilde y_i = z_{\text{wheel},i} - h_{\text{wheel},i}(\hat{\mathbf x}^-)$
   and their NIS are computed against the same state $\hat{\mathbf x}^-$, before any wheel updates the state.
   These are the published residuals, and they decide which wheels fail the gate.
2. **Update.** The wheels not left out (see below) then update the state one after another, each with its
   innovation recomputed against the current state.

Without the first pass, the processing order would distort the residuals. Each update pulls the state
toward that wheel's reading, so later wheels are compared with a state that earlier wheels have
already moved. If wheel 1 were stuck and processed first, it would pull $v$ toward 0; the healthy
wheels 2–6 would then all show innovations, and the fault would be spread over all six. With one
common prediction, the stuck wheel shows a large residual and the healthy ones show small residuals,
whatever the order.

Of the wheels that fail the gate, only the worst `max_gated_wheels` = 2 (by NIS) are left out of the
update; the others still update the state:
- One or two failing wheels are most likely faulty, for example stuck. Leaving them out of the update
  keeps them from dragging the state.
- When more fail at once, the likelier cause is a wrong state (for example at start-up). The remaining
  failing wheels update the state and pull it back.
- An earlier rule, "if all fail, use all to update the state", locked a stuck wheel into a compromise state.

**Side slip: no limit on wheels left out of the update.** Each wheel's constraint
$z_{\text{sideslip},i} = 0$ is gated on its own. Every wheel that fails the gate is left out of the
update, even if all 6 fail (`max_excluded` is 6 here, against 2 for the rolling speeds). Its innovation
is still published. The rolling-speed limit is not needed here, for two reasons:
- **The constraint, not the state, is the likely error.** Side slip is a pseudo-measurement: it
  assumes the wheel does not slide along its axle. A wheel that fails the gate is one where that
  assumption does not hold, because the wheel is really sliding or its steer angle does not match the
  motion (for example a stuck corner). Updating the state with it would force false information into
  $v$ and $v_{\text{lat}}$.
- **It is not needed to correct a wrong state.** The rolling-speed updates run just before, in the same
  step, and have already corrected $v$ and $v_{\text{lat}}$. If the whole rover really slides sideways,
  all 6 constraints fail and none updates the state; $v_{\text{lat}}$ then follows the lateral
  accelerometer, which is correct during a slide. The large side-slip innovations on all wheels are
  the published sign of the slide.

**Gyro and heading: reacquisition.** These channels use the gate plus a recovery rule, in
`_update_channel()`.

*The problem: gate lockout.* The gate works for a single outlier, but fails when the offset between
reading and prediction persists. Take a gyro whose bias suddenly steps up (an injected fault):
1. The reading is now offset from $h_{\text{gyro}} = \omega + b_g$, so its NIS exceeds the gate and it
   does not update the state.
2. Since the state is not updated, $\hat b_g$ never moves toward the new bias.
3. The next reading is therefore offset by the same amount, and also fails the gate.

The channel would stay frozen forever: the filter stops using the gyro, and the published residual
stays large even after the fault has cleared.

*The rule.* For each gyro or heading reading:
1. Compute $\tilde y$ and NIS. If $\text{NIS} \le 10.828$, update the state with it and reset the
   channel's failure timer.
2. If $\text{NIS} > 10.828$, the reading does not update the state. Start the failure timer if it is
   not already running. The innovation is still published.
3. If the channel's readings have failed the gate **continuously** for `reacquire_s` = 3 s (90 steps at 30 Hz):
   - add $\tilde y^2$ to the variance of the channel's own state, $P_{jj} \mathrel{+}= \tilde y^2$,
     where $j$ is $b_g$ for the gyro and $\psi$ for the heading;
   - recompute the innovation with this inflated $P$, and update the state with it, bypassing the gate;
   - reset the failure timer and increment the counter `/Rover/estimator/reacquisitions`.

*Why these choices:*
- **Only the channel's own state is inflated.** A persistent gyro offset most likely means the gyro
  bias changed, so $b_g$ should absorb it. A persistent heading offset means $\psi$ itself is wrong.
  Inflating only that state directs the correction there, not into $\omega$ or $v$.
- **The amount is $\tilde y^2$.** It makes the state's standard deviation at least $|\tilde y|$, so the
  needed jump is at most about 1σ. The gain on that state is then $K_j \approx P_{jj}/(P_{jj} + R) \approx 1$,
  and one update moves the state most of the way to the new value.
- **The wait is 3 s.** A single outlier or a short spike, for example from a bump, clears within 3 s
  and those readings just fail the gate, leaving the state unchanged. A real step lasts longer and is absorbed.

*What a gyro bias fault looks like:* at onset, a large spike in the gyro innovation and NIS lasting
about 3 s; then reacquisition: $\hat b_g$ jumps to about the injected offset, the innovation returns to
normal, and the reacquisition counter increments. The fault signature is this combination: an
innovation spike, a jump in `bias.gyro`, and a counter increment.

## Monitor-only residuals

These are evaluated against the EKF estimate but **never update the state**. If motor effort updated
the state, the filter would explain a slipping wheel (too little torque) as accelerometer bias, and
the residual would vanish while the fault persisted (derived in
[Why the torque does not update the state](#why-the-torque-does-not-update-the-state)).

**Lateral force.** The sideways specific force the estimated motion does not explain:

```math
e_{\text{lat}} = \frac{a_l - g\sin\phi\cos\theta - \hat\omega\,\hat v}{\sigma_a}
```

With grip, the wheels supply exactly the force that cancels cross-slope gravity and turns the rover,
so this is zero-mean. When grip is lost, the rover slides downhill on a cross-slope or understeers in
a turn, and the shortfall persists while it slides. Steady sliding on flat ground needs no force, so
it is invisible here; physically it does not happen.

**Drive torque (per wheel).** `motor_effort` is now the solver's measured joint force
(`Articulation.get_measured_joint_efforts`); see [Motor effort](README.md#motor-effort-what-it-measures). The
wheel must push its share of the rover through the forward specific force (acceleration plus the
slope's pull, which is exactly what $a_f$ measures), plus resistances. The model is linear in its
coefficients:

```math
\hat\tau_i = c_{1,i}\,(a_f - \hat b_a) + c_{2,i}\tanh\!\Big(\frac{\dot\theta_i}{0.05}\Big) + c_{3,i}\,\dot\theta_i + c_{4,i}\,\ddot\theta_i + c_{5,i}\,|\hat\omega|
```

The terms are, in order: load ($c_1 \approx r\,m_i$), rolling resistance, viscous loss, wheel
inertia, and steering scrub.
- $\ddot\theta_i$ is an exponential moving average (weight 0.2) of the finite-difference joint
  acceleration.
- The coefficients are fitted by least squares on nominal data. The fitted $c_1$ values sum to a
  plausible ~560 kg, with the rear wheels carrying the most.

The residual is the **excess torque in the direction of motion**:

```math
e_{\tau,i} = \operatorname{sign}(\dot\theta_{\text{cmd},i})\ \frac{\tau_i - \hat\tau_i}{\sigma_{\tau,i}}
```

$e_\tau < 0$ means the wheel spins too easily (slip); $e_\tau > 0$ means too much resistance (sink,
stuck). The sign term keeps that meaning when reversing.

**Command tracking (per wheel).** The shortfall of the wheel rate against what the controller
commanded:

```math
e_{\text{trk},i} = \operatorname{sign}(\dot\theta_{\text{cmd},i})\ \frac{\dot\theta_{\text{cmd},i} - \dot\theta_i}{\sigma_{\text{trk}}}
\qquad \big(\text{if } \dot\theta_{\text{cmd}} = 0:\ -|\dot\theta_i| / \sigma_{\text{trk}}\big)
```

A velocity-controlled wheel that slips still tracks its command ($\approx 0$). One that is sunk,
stuck or torque-limited cannot ($> 0$). The command is stored by `drive_controller._apply()` as
`last_wheel_command`.

### Why the torque does not update the state

Suppose wheel $i$'s torque $\tau_i$ were used as a scalar measurement. Its model is $\hat\tau_i$, and
$b_a$ is the state in it that can absorb an error. Write the other terms as $\tau_{\text{rest},i}$
(the $|\hat\omega|$ term would leak into $\omega$ the same way; it is left out for clarity):

```math
h_\tau(\mathbf x) = c_{1,i}\,(a_f - b_a) + \tau_{\text{rest},i}, \qquad
H_{\tau,b_a} = \frac{\partial h_\tau}{\partial b_a} = -c_{1,i}
```

**1. Fault onset.** A slipping wheel needs $\Delta\tau > 0$ less torque than the model predicts.
With $\hat b_a = b_a$:

```math
\tau_i = c_{1,i}\,(a_f - b_a) + \tau_{\text{rest},i} - \Delta\tau
\quad\Rightarrow\quad
\tilde y_0 = \tau_i - h_\tau(\hat{\mathbf x}) = -\Delta\tau
```

**2. One update**, with $P = P_{b_a b_a}$ and $R_\tau$ the torque noise variance:

```math
S = c_{1,i}^2 P + R_\tau, \qquad
K_{b_a} = \frac{P\,H_{\tau,b_a}}{S} = -\frac{c_{1,i} P}{S}, \qquad
\hat b_a^+ = \hat b_a + K_{b_a}\tilde y_0 = \hat b_a + \frac{c_{1,i} P}{S}\,\Delta\tau
```

**3. The innovation after the update:**

```math
h_\tau(\hat{\mathbf x}^+) = h_\tau(\hat{\mathbf x}) - c_{1,i}\,(\hat b_a^+ - \hat b_a)
= h_\tau(\hat{\mathbf x}) - \frac{c_{1,i}^2 P}{S}\,\Delta\tau
\quad\Rightarrow\quad
\tilde y_1 = -\Delta\tau + \frac{c_{1,i}^2 P}{S}\,\Delta\tau = -\Delta\tau\,\frac{R_\tau}{S}
```

Since $R_\tau/S < 1$, every update shrinks the innovation by that factor. $P$ cannot collapse to 0,
because the random walk adds $q_{b_a}\Delta t$ every step, so

```math
\tilde y_n = -\Delta\tau \prod_{k=1}^{n} \frac{R_\tau}{S_k} \;\to\; 0, \qquad
\hat b_a \to b_a + \frac{\Delta\tau}{c_{1,i}}, \qquad
e_{\tau,i} \to 0 \quad \text{while } \Delta\tau > 0
```

**4. The error spreads to $v$.** The prediction uses $a_f - \hat b_a$, so:

```math
\Delta b_a = \hat b_a - b_a = \frac{\Delta\tau}{c_{1,i}}, \qquad
\Delta\dot v = -\Delta b_a = -\frac{\Delta\tau}{c_{1,i}}
\quad\Rightarrow\quad
\Delta v(t) \approx -\frac{\Delta\tau}{c_{1,i}}\,t
```

```math
\tilde y_{\text{wheel},j} \approx -\cos\delta_j\,\Delta v \quad \text{on all wheels } j
```

The rolling speeds pull $\hat b_a$ back through $F_{v,b_a}$, so the two channels fight. Either way,
the published `bias.accel` moves by up to $\Delta\tau/c_{1,i}$: the signature of an accelerometer bias
fault, not of a wheel fault.

**Without the update (the actual design)**, $\hat b_a$ is unaffected, so

```math
e_{\tau,i} = -\frac{\Delta\tau}{\sigma_{\tau,i}} \quad \text{for as long as the slip lasts}
```

constant, negative, and on wheel $i$ only.

## From residuals to alarms

Every residual is non-zero on every sample, because of sensor noise. A fault is a **lasting shift in
the mean**; noise averages out.

**1 s windows.** The downlink is 1 Hz and the filter 30 Hz. A single sample per second would miss
transients and carry the full per-sample noise. So each window ($n \approx 30$ samples) reports two
values per residual:

```math
\bar e = \frac1n\sum_{k} e_k \quad(\text{mean}), \qquad e^{\max} = e_{k^*},\ k^* = \arg\max_k |e_k| \quad(\text{signed extreme})
```

Windows close on simulation time inside the filter, independent of when the downlink asks. Each
window also reports the mean NIS per channel; for the wheels, that is the NIS averaged over the six
rolling innovations.

**Two-sided CUSUM per residual**, updated once per window:

```math
z_w = \frac{\bar e_w}{\sigma_w},\qquad
S^+_w = \max(0,\ S^+_{w-1} + z_w - \kappa),\qquad
S^-_w = \max(0,\ S^-_{w-1} - z_w - \kappa),\qquad
\text{alarm} \iff \max(S^+_w, S^-_w) > h
```

- $\sigma_w$ = `cusum.window_sigma[key]` is the standard deviation of that residual's window mean on
  nominal data. It is not $1/\sqrt{30}$, because real residuals are correlated in time.
- $\kappa$ = `drift` = 0.5 is the smallest shift of interest, in units of $\sigma_w$.
- $h$ = `cusum.thresholds[key]` is fitted per residual: $h = \max(10,\ 1.5\cdot\max_{\text{nominal}} S)$.
  Effort and tracking carry slow terrain-dependent offsets that a single white-noise threshold
  reads as faults. With the fitted thresholds, a held-out nominal episode raises no alarm.
- Running the CUSUM on the raw 30 Hz samples instead would false-alarm every ~15 s.
- For reference, the Siegmund approximation of the mean run length of a one-sided CUSUM with a true
  shift $\mu$ (`monitor.cusum_mean_run_length`) is

  ```math
  \text{ARL}(\mu) \approx \frac{e^{-2\Delta b} + 2\Delta b - 1}{2\Delta^2},\qquad \Delta = \mu - \kappa,\quad b = h + 1.166
  ```

  With $\kappa = 0.5$ and $h = 10$ this gives about 20 h between false alarms per residual on white
  noise.

The alarms are packed into `cusum_alarm`, one bit per residual in `nav_filter.RESIDUAL_KEYS` order:

| Bits | Residuals (each in wheel order FL, FR, ML, MR, RL, RR) |
|---|---|
| 0–5 | `wheel_rolling` |
| 6–11 | `wheel_sideslip` |
| 12–17 | `wheel_effort` |
| 18–23 | `wheel_tracking` |
| 24, 25, 26 | `gyro`, `heading`, `lateral_force` |

> **Alarms detect; magnitudes isolate.** A stuck wheel drags the rover, so its neighbours scrub and
> their CUSUMs fire too. In the stuck-FL run the FL rolling mean was −7.7σ while the others stayed
> within ±2.4σ. Identify the faulty wheel by the largest window mean, not by which bits are set.

## Telemetry: `/Rover/estimator`

| Parameter | Content |
|---|---|
| `pose` {x, y, yaw} | estimated episode-local position (m) and heading (deg) |
| `motion` {speed, lateral_speed, yaw_rate} | $\hat v$, $\hat v_{\text{lat}}$ (m/s), $\hat\omega$ (rad/s) |
| `bias` {gyro, accel} | $\hat b_g$ (rad/s), $\hat b_a$ (m/s²) |
| `imu_residual` {gyro, heading, lateral_force} × {_mean, _max} | normalized, σ |
| `nis` {wheels, gyro, heading} | window mean NIS; nominally ≈ 1 |
| `wheel_rolling_{mean,max}` [6] | $\tilde y_i/\sqrt{S_i}$, rolling speed |
| `wheel_sideslip_{mean,max}` [6] | $\tilde y_i/\sqrt{S_i}$, side slip |
| `wheel_effort_{mean,max}` [6] | $e_{\tau,i}$ (absent until the torque model is fitted) |
| `wheel_tracking_{mean,max}` [6] | $e_{\text{trk},i}$ |
| `cusum_alarm` | uint32 bitmask, above |
| `reacquisitions` | episode count of gyro/heading reacquisitions |

- Wheel arrays are in the order front_left, front_right, mid_left, mid_right, rear_left, rear_right.
  That differs from `motor_encoder`, which lists the left side first.
- In the dataset these are observable columns such as `estimator.wheel_rolling_mean.0`, since they
  are computed from the rover's own sensors.
- A field the filter did not produce is *skipped*, never zero-filled, so missing reads as missing.
- `visualize_dataset.py` plots the rolling, torque, tracking and IMU residuals in the bottom two rows
  of each episode page.

## Fitting the parameters

Every number in `estimator:` in `cfg/robot/perseverance.yaml` is fitted from nominal driving. To refit
after changing the rover, terrain or physics:

```bash
# in the container: nominal episodes only (zero every class weight except nominal in a copy of the
# robot config and pass it with --robot-cfg), dumping the raw 30 Hz filter inputs beside the truth
/isaac-sim/python.sh /workspace/omnilrs/my_files/src/run_perseverance.py --headless --no-images \
    --robot-cfg <nominal-only copy> --dataset-out my_files/datasets/nav_cal --episodes 4 \
    --estimator-dump my_files/datasets/nav_cal_dump

# on the host
python3 /home/yifan/git/OmniLRS/scripts/fit_nav_estimator.py ~/docker/isaac-sim/my_files/datasets/nav_cal_dump \
    --out estimator_fit.yaml     # then paste the block into cfg/robot/perseverance.yaml
```

The script runs these steps (truth is used only here, offline):

1. **Truth kinematics.** From the base-link pose: $\psi$, $\theta$, $\phi$ via the body axes;
   $v = \dot{\mathbf p}\cdot\mathbf f_w$ and $v_{\text{lat}} = \dot{\mathbf p}\cdot\mathbf l_w$;
   $\omega = \dot\psi$. All derivatives are smoothed over 0.3 s.
2. **IMU mounting.** Search the 24 signed axis pairs $(\mathbf f, \mathbf l)$. Keep the one whose
   derived heading minimizes $\operatorname{median}|\psi_{\text{imu}} - \psi|$. Then report the
   correlations of $\theta$, $\phi$, $a_f$, $a_l$ and $\omega_{\text{up}}$ with their truth counterparts.
3. **Wheel yaw scale.** Solve the per-sample least-squares twist from the 12 wheel equations,
   $(v, v_{\text{lat}}, \omega_{\text{wh}})$. Then, over the turning samples ($|\omega_{\text{up}}| > 0.05$ rad/s),
   $k = \sum \omega_{\text{wh}}\, \omega_{\text{up}} \,/\, \sum \omega_{\text{up}}^2$.
4. **Sensor noise (R)**, all robust against outliers ($\text{MAD} \times 1.4826$):
   - $\sigma_{\text{wheel}}$: of $z_{\text{wheel},i} - h_{\text{wheel},i}(\mathbf x_{\text{true}})$
   - $\sigma_{\text{sideslip}}$: of $h_{\text{sideslip},i}(\mathbf x_{\text{true}})$
   - $\sigma_{\text{heading}}$: of $\psi_{\text{imu}} - \psi$

   Two more are measured without the truth, as the spread of the signal around itself:
   - $\sigma_g$ and $\sigma_a$: the std of the signal minus its 0.3 s moving average
   - $\sigma_{\text{trk}}$: the plain std of $\dot\theta_{\text{cmd}} - \dot\theta$. Not MAD, because
     the error is near zero except in the lag after each command change, and MAD makes those normal
     transients look like 20σ events.
5. **Torque model.** Least squares per wheel on the regressors above, with $\sigma_{\tau,i}$ as the
   residual std. One episode is held out; its normalized residual must have mean ≈ 0 and std ≈ 1.
6. **Consistency (Q), by replay.** The filter replays every dump, and three parameters are tuned from
   the channels' mean NIS. R is treated as a sensor property and Q as a model property:

   ```math
   \sigma_{\text{wheel}} \leftarrow \sigma_{\text{wheel}}\,\text{NIS}_{\text{wh}}^{1/4},\qquad
   q_\omega \leftarrow q_\omega\,\text{NIS}_{g}^{1/2},\qquad
   q_\psi \leftarrow q_\psi\,\text{NIS}_{\psi}^{1/2}
   ```

   - These are half steps in log space (full steps oscillated), clipped, and repeated until every NIS
     is in [0.8, 1.25] or 8 passes have run.
   - The gyro and heading σ keep their measured values. Inflating them instead (tried first) hid the
     model errors and blunted both sensors.
7. **CUSUM.** $\sigma_w$ is the std of each residual's nominal window means. $h$ is 1.5× the largest
   nominal CUSUM level, as above.

**Current fit** (4 nominal 600 s episodes, 72 000 samples):

| Quantity | Value |
|---|---|
| Replay NIS | wheels 1.23, gyro 1.03, heading 0.87 |
| $\sigma_{\text{wheel}}$, $\sigma_{\text{sideslip}}$ | 0.0156 m/s, 0.0047 m/s |
| $\sigma_g$, $\sigma_\psi$ | 0.0084 rad/s, 0.00066 rad (0.04°) |
| $\sigma_a$ | 0.40 m/s² (chassis vibration) |
| $q_\omega$, $q_\psi$ | 0.0024 (rad/s)²/s, 1.4·10⁻⁵ rad²/s |
| $k$ (`wheel_yaw_scale`) | 1.034 |
| Torque model $R^2$ per 30 Hz sample | 0.09–0.23 (usable only through the windows) |

## Validation in Isaac Sim

Single-fault runs: 300 s each, fault onset at 150 s, fitted config, dataset seed 41 (not used for
fitting). "Detected" is the first new CUSUM alarm after the onset; the σ values are post-onset window
means.

| Fault | Detected after | Signature |
|---|---|---|
| none (held out) | no alarm | all window means within ±0.3σ; $\hat b_g$ = 0.0002 rad/s |
| `wheel_stuck:front_left:0.8` | 1 s | FL rolling −7.7σ, FL tracking +16.5σ |
| `steer_stuck:front_left:30` | 4 s | FL side slip +5.6σ, FL rolling +2.9σ |
| `wheel_sink:ALL:0.8` | 6 s | effort +0.6 to +1.7σ on every wheel |
| `imu:0.5:0` | 1 s | $\hat b_g$ → 0.10 rad/s, heading −20σ, side slip −24σ on all wheels, lateral force +2.2σ |
| `wheel_slip:ALL:0.8` | 20 s | weak shifts only (rolling ±0.8σ, effort up to +0.6σ) |
| `wheel_slip:front_left:0.8` | **not detected** | — |

What these runs show:
- **One-wheel slip is invisible to these sensors.** The velocity drive keeps the slipping wheel on
  its commanded rate, so its spin rate sensor, tracking and scrub all look normal. The only trace is less
  torque, and it is below the torque model's noise. Seeing it needs a ground-relative speed
  measurement such as visual odometry.
- **An IMU fault also disturbs the wheel residuals**, because the injector biases the orientation
  (all three angles) and the accelerometer, not only the gyro. The biased attitude corrupts gravity
  removal. The filter has no state for the attitude bias (see
  [State and process model](#state-and-process-model-prediction)). Its unique signature is the jump
  in $\hat b_g$.
- **Small validation:** one episode per fault at one severity. The detection times are indicative.
