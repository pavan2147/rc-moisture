\begin{algorithm}
\caption{Step-by-step algorithm for  Physics-Informed Neural Networks (PINNs)}
\begin{algorithmic}[1]
\Statex \textbf{Inputs:}
\Statex \quad Neural network $\bar\theta(z, t; \Theta)$ with parameters $\Theta$
\Statex \quad Initial condition: $\theta(z, 0) = \frac{1}{2} + \frac{1}{2} \tanh\left( -\frac{z}{4} \right)$
\Statex \quad Domain: $z \in [0, 5]$, $t \in [0, 10]$
\Statex \quad Number of initial condition points: $N_{\theta} = 1500$
\Statex \quad Number of collocation points: $N_{f} = 7000$
\Statex \quad Number of training iterations: $N_{iter} = 50000$
\Statex \quad Optimizer: Adam (learning rate = 0.001)
\Statex \textbf{Output:}
\Statex \quad Optimized network parameters $\Theta^*$

\State \textbf{Step 0: Initialization}
\State Initialize network parameters $\Theta$
\State Initialize optimizer state
\State Set iteration counter $k \gets 0$

\Function{PDEResidual}{$z, t, \bar\theta, \Theta$}
    \State Compute network output: $\bar\theta(z, t; \Theta)$
    \State Compute derivatives:
    \State \quad $\bar\theta_t \gets \frac{\partial \bar\theta}{\partial t}$, $\bar\theta_z \gets \frac{\partial \bar\theta}{\partial z}$, $\bar\theta_{zz} \gets \frac{\partial^2 \bar\theta}{\partial z^2}$
    \State Compute residual: $f(z,t) \gets \bar\theta_t + \bar\theta \bar\theta_z - \bar\theta_{zz}$
    \State \Return $f(z,t)$
\EndFunction

\While{$k < N_{iter}$}
    \State \textbf{Step 1.1: Initial condition loss}
    \State Sample $\{z_{\theta}^i\}_{i=1}^{N_{\theta}} \sim \mathcal{U}(0, 5)$
    \State Set $\{t_{\theta}^i\}_{i=1}^{N_{\theta}} \gets 0$
    \State Compute true initial values: $\theta^i = \theta(z_{\theta}^i, 0)$
    \State Predict values: $\bar\theta(z_{\theta}^i, t_{\theta}^i; \Theta)$
    \State Compute initial loss: $\text{MSE}_{\theta} = \frac{1}{N_{\theta}} \sum_{i=1}^{N_{\theta}} \left( \bar\theta(z_{\theta}^i, t_{\theta}^i; \Theta) - \theta^i \right)^2$

    \State \textbf{Step 1.2: PDE residual loss}
    \State Sample $\{(z_f^i, t_f^i)\}_{i=1}^{N_f} \sim \mathcal{U}([0,5] \times [0,10])$
    \State Compute PDE residuals: $f(z_f^i, t_f^i) = \text{PDEResidual}(z_f^i, t_f^i, \bar\theta, \Theta)$
    \State Compute residual loss: $\text{MSE}_{f} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left( f(z_f^i, t_f^i) \right)^2$

    \State \textbf{Step 1.3: Total loss and parameter update}
    \State Compute total loss: $j(\Theta) = \text{MSE}_{\theta} + \text{MSE}_{f}$
    \State Compute gradient $\nabla_{\Theta} j(\Theta)$
    \State Update parameters: $\Theta \gets \text{OptimizerStep}(\Theta, \nabla_{\Theta} j(\Theta))$

    \State Increment iteration counter: $k \gets k + 1$
\EndWhile

\State \textbf{Step 2: Return trained network}
\State \Return $\Theta^*$
\end{algorithmic}
\label{alg:richards_pinn}
\end{algorithm}
