# Repilot Patch

```
                if (hasExp
                    || hasDecPoint) {
                    return false;
                }
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

@@ -1410,7 +1410,11 @@

             if (chars[i] == 'l'
                 || chars[i] == 'L') {
                 // not allowing L with an exponent or decimal point
-                return foundDigit && !hasExp;
+                if (hasExp
+                    || hasDecPoint) {
+                    return false;
+                }
+                return foundDigit;
             }
             // last character is illegal
             return false;
```

# Note

