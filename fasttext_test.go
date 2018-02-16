package fasttextgo

import (
	"fmt"
	"testing"
)

func TestSimple(t *testing.T) {
	LoadModel("/Users/jason/datasets/bins/lid.176.bin")
	tags, err := Predict("hello this is a new day and we are new man bueno", 6)
	if err != nil {
		panic(err)
	}
	for _, t := range tags {
		fmt.Println(t.Label, t.Probability)
	}
	// fmt.Println(Predict("sushi in palo alto"))
	// fmt.Println(Predict("muy caliente"))
}
