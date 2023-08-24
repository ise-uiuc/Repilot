# Repilot Patch

```
                if (val1 == Long.MIN_VALUE) {
                    throw new ArithmeticException("Multiplication overflows a long: " + val1 + " * -1");
                }
```

# Developer Patch

```
                if (val1 == Long.MIN_VALUE) {
                    throw new ArithmeticException("Multiplication overflows a long: " + val1 + " * " + val2);
                }
```

# Context

```
--- bug/Time-15/src/main/java/org/joda/time/field/FieldUtils.java

+++ fix/Time-15/src/main/java/org/joda/time/field/FieldUtils.java

@@ -135,6 +135,9 @@

     public static long safeMultiply(long val1, int val2) {
         switch (val2) {
             case -1:
+                if (val1 == Long.MIN_VALUE) {
+                    throw new ArithmeticException("Multiplication overflows a long: " + val1 + " * -1");
+                }
                 return -val1;
             case 0:
                 return 0L;
```

# Note

