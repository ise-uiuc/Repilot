# Repilot Patch

```
return foundDigit && !hasExp && !hasDecPoint;
```

# Developer Patch

```
                return foundDigit && !hasExp && !hasDecPoint;
```

# Context

```
--- bug/Lang-24/src/main/java/org/apache/commons/lang3/math/NumberUtils.java

+++ fix/Lang-24/src/main/java/org/apache/commons/lang3/math/NumberUtils.java

@@ -1410,7 +1410,7 @@

             if (chars[i] == 'l'
                 || chars[i] == 'L') {
                 // not allowing L with an exponent or decimal point
-                return foundDigit && !hasExp;
+return foundDigit && !hasExp && !hasDecPoint;
             }
             // last character is illegal
             return false;
```

# Note

