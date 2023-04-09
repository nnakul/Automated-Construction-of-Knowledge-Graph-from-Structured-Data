exec:
	@g++-11 source.cpp -o exec

create:
	@echo "-1\n-1\n-1" > myKG.txt

reset:
	@echo "-1\n-1\n-1\n" > myKG.txt

show:
	@echo ""
	@cat myKG.txt

add: exec
	./exec myKG.txt $(db)

clean:
	@rm exec