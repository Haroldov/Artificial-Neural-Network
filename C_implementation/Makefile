CC = gcc
SRC = main.c
OBJ = $(SRC:%.c=%.o)
LIB = -Wl,-rpath=/usr/local/lib
LIBFLAGS = -lgsl -lgslcblas -lm
INLINEMACRO = -DHAVE_INLINE
NAME = nn
RM = rm
CFLAGS = -Wall -Werror -Wextra -pedantic

%.o: %(SRC)
	$(CC) -c -o $@ $^ $(CFLAGS)

all: $(OBJ)
	$(CC) $(LIB) $(OBJ) -o $(NAME) $(INLINEMACRO) $(LIBFLAGS)

clean:
	$(RM) -f *~ $(NAME)

oclean:
	$(RM) -f $(OBJ)

fclean: clean oclean

re: oclean all
