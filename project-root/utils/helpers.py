# Returns the appropriate device (cuda or cpu)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Additional utility functions added here