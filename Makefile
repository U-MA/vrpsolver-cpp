#Set this to @ to keep the makefile quiet
SILENCE = @

#---- Outputs ----#
COMPONENT_NAME = VrpSolver-cpp
TARGET_LIB = \
	lib/lib$(COMPONENT_NAME).a
	
TEST_TARGET = \
	$(COMPONENT_NAME)_tests

#--- Inputs ----#
PROJECT_HOME_DIR = .
CPPUTEST_HOME = cpputest-3.5
CC = gcc
CPP_PLATFORM = Gcc

SRC_DIRS = \
	$(PROJECT_HOME_DIR)/src \
	$(PROJECT_HOME_DIR)/symphony \

TEST_SRC_DIRS = \
	tests\

INCLUDE_DIRS =\
	.\
	$(CPPUTEST_HOME)/include \
	$(PROJECT_HOME_DIR)/include \
	$(PROJECT_HOME_DIR)/symphony \
  
#CPPUTEST_WARNINGFLAGS += -pedantic-errors -Wconversion -Wshadow  -Wextra
CPPUTEST_WARNINGFLAGS += -Wall -Werror -Wswitch-default -Wswitch-enum 

CXXFLAGS += -include $(CPPUTEST_HOME)/include/CppUTest/MemoryLeakDetectorNewMactos.h

LDFLAGS = -lstdc++

include $(CPPUTEST_HOME)/build/MakefileWorker.mk
