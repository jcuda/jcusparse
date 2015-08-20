/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.jcusparse;


import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Basic test of the bindings of the JCusparse class
 */
public class JCusparseBasicBindingTest
{
    public static void main(String[] args)
    {
        JCusparseBasicBindingTest test = new JCusparseBasicBindingTest();
        test.testJCusparse();
    }

    @Test
    public void testJCusparse()
    {
        assertTrue(BasicBindingTest.testBinding(JCusparse.class));
    }
    

}
