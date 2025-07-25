#!/usr/bin/env python3
"""
Read-only wrapper for environments to prevent algorithms from modifying environment state.
"""

import numpy as np
from typing import Any, Set, List, Dict
import copy


class ReadOnlyEnvironmentWrapper:
    """
    A wrapper that makes an environment read-only by preventing modifications to its state.
    
    This wrapper:
    1. Intercepts all attribute assignments and prevents them
    2. Returns deep copies of mutable objects to prevent indirect modification
    3. Allows only read operations on the wrapped environment
    """
    
    def __init__(self, environment):
        """
        Initialize the read-only wrapper.
        
        Args:
            environment: The environment instance to wrap
        """
        # Use object.__setattr__ to bypass our own __setattr__ restriction
        object.__setattr__(self, '_wrapped_env', environment)
        object.__setattr__(self, '_readonly_mode', True)
        
        # Store original methods that should be read-only
        object.__setattr__(self, '_read_only_methods', {
            'get_available_arms_for_round',
            'get_feasible_combinations', 
            'get_expected_reward_for_path',
            'get_optimal_path_expected_reward',
            'get_reward_for_round',
            'cleanup'
        })
        
        # Methods that should NOT be allowed (they modify state)
        object.__setattr__(self, '_forbidden_methods', {
            '__setattr__',
            '__setitem__',
            '__delattr__',
            '__delitem__'
        })
    
    def __setattr__(self, name, value):
        """Prevent any attribute modifications."""
        if hasattr(self, '_readonly_mode') and self._readonly_mode:
            raise AttributeError(f"Cannot modify attribute '{name}' in read-only environment")
        object.__setattr__(self, name, value)
    
    def __delattr__(self, name):
        """Prevent any attribute deletions."""
        raise AttributeError(f"Cannot delete attribute '{name}' in read-only environment")
    
    def __getattr__(self, name):
        """
        Delegate attribute access to wrapped environment, with protection for mutable objects.
        """
        if name.startswith('_'):
            # Internal attributes
            return object.__getattribute__(self, name)
        
        if name in self._forbidden_methods:
            raise AttributeError(f"Method '{name}' is not allowed in read-only environment")
        
        # Get attribute from wrapped environment
        attr = getattr(self._wrapped_env, name)
        
        # If it's a method, return it as-is (methods are immutable)
        if callable(attr):
            if name in self._read_only_methods:
                # Wrap methods that return mutable objects
                def wrapped_method(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    return self._make_readonly(result)
                return wrapped_method
            else:
                # Allow other methods but warn if they might modify state
                if hasattr(attr, '__self__') and not name.startswith('get_') and not name.startswith('is_'):
                    print(f"Warning: Calling method '{name}' which might modify environment state")
                return attr
        
        # For non-callable attributes, return read-only versions
        return self._make_readonly(attr)
    
    def _make_readonly(self, obj):
        """
        Make an object read-only by returning copies of mutable types.
        """
        if isinstance(obj, (list, set)):
            # Return a copy to prevent modification of the original
            return copy.deepcopy(obj)
        elif isinstance(obj, dict):
            # Return a copy to prevent modification of the original
            return copy.deepcopy(obj)
        elif isinstance(obj, np.ndarray):
            # Return a read-only view of numpy arrays
            readonly_array = obj.view()
            readonly_array.flags.writeable = False
            return readonly_array
        else:
            # Immutable types (int, float, str, tuple) can be returned as-is
            return obj
    
    def __repr__(self):
        return f"ReadOnlyEnvironmentWrapper({repr(self._wrapped_env)})"
    
    def __str__(self):
        return f"ReadOnly({str(self._wrapped_env)})" 