exec.exe: source.cpp
	@g++-11 source.cpp -o exec.exe

create:
	@echo "-1\n-1\n-1" > myKG.txt

reset:
	@echo "-1\n-1\n-1\n" > myKG.txt

show:
	@echo ""
	@cat myKG.txt
	@echo ""

add: exec.exe
	./exec.exe myKG.txt data/$(db)

clean:
	@rm *.exe

turt.exe: toturtle.cpp
	@g++-11 toturtle.cpp -o turt.exe

turtle:	turt.exe
	./turt.exe < myKG.txt > myKG.ttl