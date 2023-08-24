# Repilot Patch

```
return false;
```

# Developer Patch

```
return false;
```

# Context

```
--- bug/Lang-51/src/java/org/apache/commons/lang/BooleanUtils.java

+++ fix/Lang-51/src/java/org/apache/commons/lang/BooleanUtils.java

@@ -679,6 +679,8 @@

                         (str.charAt(1) == 'E' || str.charAt(1) == 'e') &&
                         (str.charAt(2) == 'S' || str.charAt(2) == 's');
                 }
+
+                return false;
             }
             case 4: {
                 char ch = str.charAt(0);
```

# Note

