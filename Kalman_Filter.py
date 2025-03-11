import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = A  # State transition matrix
        self.B = B  # Control input matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state estimate
        self.P = P0  # Initial covariance estimate

    def predict(self, u=np.zeros((1, 1))):
        # Predict the next state
        self.x = self.A @ self.x + self.B @ u
        # Predict the next covariance
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        # Compute the Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        # Update the state estimate
        self.x = self.x + K @ (z - self.H @ self.x)
        # Update the covariance estimate
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P

    def get_state(self):
        return self.x

# Simulation parameters
dt = 2  # Time step
A = np.array([[1, dt], [0, 1]])  # State transition matrix
B = np.array([[0.5 * dt**2], [dt]])  # Control input matrix
H = np.array([[1, 0]])  # Observation matrix
Q = np.array([[1, 0], [0, 3]])  # Process noise covariance
R = np.array([[10]])  # Measurement noise covariance
x0 = np.array([[0], [1]])  # Initial state estimate
P0 = np.array([[1, 0], [0, 1]])  # Initial covariance estimate

# Generate synthetic data
n_steps = 50
true_positions = []
measurements = []
x_true = np.array([[0], [1]])  # Initial true state
for _ in range(n_steps):
    u = np.zeros((1, 1))  # Control input (zero in this case)
    x_true = A @ x_true + B @ u  # True state
    z = H @ x_true + np.random.normal(0, np.sqrt(R[0, 0]), (1, 1))  # Measurement
    true_positions.append(x_true[0, 0])
    measurements.append(z[0, 0])

# Apply Kalman filter
kf = KalmanFilter(A, B, H, Q, R, x0, P0)
estimates = []
for z in measurements:
    kf.predict()
    kf.update(np.array([[z]]))
    estimates.append(kf.get_state()[0, 0])

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(true_positions, label='True Position')
plt.plot(measurements, label='Measurements', linestyle='dotted')
plt.plot(estimates, label='Kalman Filter Estimate', linestyle='dashed')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Kalman Filter Tracking')
plt.show()
