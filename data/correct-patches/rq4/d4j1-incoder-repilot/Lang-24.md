# Repilot Patch

```
                if (hasDecPoint || hasExp) {
                    // two decimal points or dec in exponent
                    return false;
                }
                // single trailing L after non-exponent is ok
                return foundDigit;
```

# Developer Patch

```
                return foundDigit && !hasExp && !hasDecPoint;
```

# Context

```
--- bug/Lang-24/src/main/java/org/apache/commons/lang3/math/NumberUtils.java

+++ fix/Lang-24/src/main/java/org/apache/commons/lang3/math/NumberUtils.java

@@ -1410,7 +1410,12 @@

             if (chars[i] == 'l'
                 || chars[i] == 'L') {
                 // not allowing L with an exponent or decimal point
-                return foundDigit && !hasExp;
+                if (hasDecPoint || hasExp) {
+                    // two decimal points or dec in exponent
+                    return false;
+                }
+                // single trailing L after non-exponent is ok
+                return foundDigit;
             }
             // last character is illegal
             return false;
```

# Note

