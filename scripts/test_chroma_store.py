from src.vector_store import get_vector_store


def main():
    store = get_vector_store()

    print("Chroma initialized successfully")
    print("Total vectors:", store.count())


if __name__ == "__main__":
    main()