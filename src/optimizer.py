"""A simple optimizer module.

This module provides a basic implementation of a gradient descent optimizer.
"""

import numpy as np
from typing import Callable

class Optimizer:
    """A class that encapsulates gradient descent optimization logic."""
    def __init__(self) -> None:
        """Initializes the Optimizer."""
        pass

    def step(
        self,
        x_start: np.ndarray,
        d_fun: Callable[[np.ndarray], np.ndarray],
        rate: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Performs a single step of gradient descent.

        Args:
            x_start (np.ndarray): The starting point for the step.
            d_fun (Callable): The function that computes the gradient.
            rate (float, optional): The learning rate. Defaults to 0.1.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the new point and
            the gradient at the original point.
        """
        grad = d_fun(x_start)
        x_new = x_start - rate * grad
        return x_new, grad

    def print_step(
        self,
        step: int,
        x_current: np.ndarray,
        grad_current: np.ndarray,
        rate: float,
    ) -> None:
        """Prints the details of a gradient descent step.

        Args:
            step (int): The current step number.
            x_current (np.ndarray): The current position.
            grad_current (np.ndarray): The current gradient.
            rate (float): The current learning rate.
        """
        x_current_str = np.array2string(x_current, precision=2, suppress_small=True)
        grad_current_str = np.array2string(grad_current, precision=2)
        print(f"Step {step}, x: {x_current_str}, rate: {rate}, grad: {grad_current_str}")

    def grad_descent(
        self,
        x_start: np.ndarray,
        d_fun: Callable[[np.ndarray], np.ndarray],
        rate: float,
        max_steps: int = 50,
        tol: float = 1e-8,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Performs gradient descent to find a minimum of a function.

        This method iteratively moves towards the minimum of a function by
        taking steps in the opposite direction of the gradient.

        Args:
            x_start (np.ndarray): The starting point for the optimization.
            d_fun (Callable): The function that computes the gradient of the
                function to be minimized.
            rate (float): The initial learning rate.
            max_steps (int, optional): The maximum number of steps to perform.
                Defaults to 50.
            tol (float, optional): The tolerance for convergence. The algorithm
                stops when the L1 norm of the difference between two
                consecutive points is less than this value. Defaults to 1e-8.

        Returns:
            tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]: A tuple
            containing:
                - The final point (approximated minimum).
                - The history of points visited during the descent.
                - The history of gradients computed at each step.
        """
        rate_current = rate
        x_current = x_start
        x_attempts = [x_start]
        grads = []
        for step in range(max_steps):
            x_current, grad_current = self.step(x_current, d_fun, rate_current)
            x_attempts.append(x_current)
            grads.append(grad_current)
            if step % 10 == 0 and step != 0:
                self.print_step(step, x_current, grad_current, rate_current)
            # A simple learning rate decay schedule
            if step % 5 == 0 and step != 0:
                rate_current = rate_current / np.sqrt(1 + 0.1 * (step % 5))
            if step > 2 and np.linalg.norm(x_current - x_attempts[-2], 1) < tol:
                print(f"Solution reached tolerance at step {step}")
                break
        return x_current, x_attempts, grads