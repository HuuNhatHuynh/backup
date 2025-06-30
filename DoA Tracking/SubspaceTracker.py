import torch 
from abc import ABC, abstractmethod


class SubspaceTracker(ABC):

    def __init__(self, beta: float, P: torch.Tensor, W: torch.Tensor) -> None:
        """
        Args:
            beta (float): forgetting factor
            P (torch.Tensor): tensor of shape (..., r, r) 
            W (torch.Tensor): tensor of shape (..., n, r)

        Returns:
            None
        """   
        self.beta = beta
        self.P = torch.clone(P)
        self.W = torch.clone(W)

    @abstractmethod
    def step(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def feed(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): tensor of shape (..., t, n)
        
        Returns:
            W (torch.Tensor): tensor of shape (..., n, r)
        """
        W_list = []
        for i in range(X.shape[-2]):
            W_list.append(self.step(X[..., i, :].unsqueeze(-1)))
        W_list = torch.stack(W_list, dim=-3)

        return W_list




class PAST(SubspaceTracker):

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor of shape (..., n, 1)
        
        Returns:
            W (torch.Tensor): tensor of shape (..., n, r)
        """
        y = self.W.conj().transpose(-2, -1) @ x
        g = y.conj().transpose(-2, -1) @ self.P / (self.beta + y.conj().transpose(-2, -1) @ self.P @ y)
        self.P = (self.P - self.P @ y @ g) / self.beta
        self.W = self.W + (x - self.W @ y) @ g

        return self.W
    


class OPAST(SubspaceTracker):
    
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor of shape (..., n, 1)
        
        Returns:
            W 
        """
        y = self.W.conj().transpose(-2, -1) @ x
        g = y.conj().transpose(-2, -1) @ self.P / (self.beta + y.conj().transpose(-2, -1) @ self.P @ y)
        self.P = (self.P - self.P @ y @ g) / self.beta
        u = (x - self.W @ y) @ g
        u_norm_squared = torch.norm(u, p=2.0, dim=[-2, -1], keepdim=True) ** 2
        alpha = (1 / torch.sqrt(1 + u_norm_squared) - 1) / u_norm_squared
        self.W = self.W + u + alpha * u_norm_squared * u + alpha * self.W @ u.conj().transpose(-2, -1) @ u 
        
        return self.W
    


# class RLS:

#     def __init__(self, beta: float, P: torch.Tensor, W: torch.Tensor) -> None:
#         """
#         Args:
#             beta (float): forgetting factor
#             P (torch.Tensor): tensor of shape (..., r, r) 
#             W (torch.Tensor): tensor of shape (..., n, r)

#         Returns:
#             None
#         """    
#         self.beta = beta
#         self.P = torch.clone(P)
#         self.W = torch.clone(W)

#     def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (torch.Tensor): tensor of shape (..., n, 1)
#             y (torch.Tensor): tensor of shape (..., r, 1)
        
#         Returns:
#             W (torch.Tensor): tensor of shape (..., n, r)
#         """
#         g = y.conj().transpose(-2, -1) @ self.P / (self.beta + y.conj().transpose(-2, -1) @ self.P @ y)
#         self.P = (self.P - self.P @ y @ g) / self.beta
#         self.W = self.W + (x - self.W @ y) @ g

#         return self.W
    


# class PASTd:

#     def __init__(self, d: torch.Tensor, W: torch.Tensor) -> None:
#         """
#         Args:
#             d (torch.Tensor): tensor of shape (..., r) 
#             W (torch.Tensor): tensor of shape (..., n, r)

#         Returns:
#             None
#         """ 
#         self.d = torch.clone(d).to(dtype=torch.complex64)
#         self.W = torch.clone(torch.linalg.qr(W)[0])

#     def update(self, X: torch.Tensor):
#         """
#         Args:
#             x (torch.Tensor): tensor of shape (..., t, n)
        
#         Returns:
#             W (torch.Tensor): tensor of shape (..., n, r)
#         """
#         for t in range(X.shape[-2]):

#             x = X[..., t, :].unsqueeze(-1)
#             beta = float((t + 1) / (t + 2))
            
#             for i in range(self.W.shape[-1]):
            
#                 y = (self.W[..., i].conj().unsqueeze(-2) @ x).squeeze(-2, -1)
#                 self.d[..., i] =  beta * self.d[..., i] + y * y.conj()
#                 e = x.squeeze(-1) - y.unsqueeze(-1) * self.W[..., i]
#                 self.W[..., i] = self.W[..., i] + (y.conj() / self.d[..., i]).unsqueeze(-1) * e
#                 x = x - y.unsqueeze(-1).unsqueeze(-1) * self.W[..., i].unsqueeze(-1)

#         self.W = torch.linalg.qr(self.W)[0]
        
#         return self.W