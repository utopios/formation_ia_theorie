from base import calculate_tanh
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calcule la tangente hyperbolique')
    parser.add_argument('values', nargs='+', type=float, 
                       help='Valeurs pour lesquelles calculer tanh')
    
    args = parser.parse_args()
    
    result = calculate_tanh(*args.values)
    print(f"tanh({args.values}) = {result}")
