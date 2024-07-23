
import subprocess

def ejecutar_identificador_movimiento():
    subprocess.call(["python", "C:\\Users\\calco\\PycharmProjects\\ProyectoFinalGestuales\\LetrasConMovimiento.py"])

def ejecutar_sin_movimiento():
    subprocess.call(["python", "C:\\Users\\calco\\PycharmProjects\\ProyectoFinalGestuales\\LetrasSinMovimiento.py"])

def main():
    print("Seleccione una opción:")
    print("1. Reconocimiento de letras con movimiento")
    print("2. Reconocimiento de letras sin movimiento")
    option = input("Ingrese el número de opción: ")

    if option == "1":
        ejecutar_identificador_movimiento()
    elif option == "2":
        ejecutar_sin_movimiento()
    else:
        print("Opción inválida. Por favor, seleccione 1 o 2.")

if __name__ == "__main__":
    main()

