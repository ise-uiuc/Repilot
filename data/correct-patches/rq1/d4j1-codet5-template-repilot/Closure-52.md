# Repilot Patch

```
return len > 0 && s.charAt(0) > '0';
```

# Developer Patch

```
    return len > 0 && s.charAt(0) != '0';
```

# Context

```
--- bug/Closure-52/src/com/google/javascript/jscomp/CodeGenerator.java

+++ fix/Closure-52/src/com/google/javascript/jscomp/CodeGenerator.java

@@ -742,7 +742,7 @@

         return false;
       }
     }
-    return len > 0;
+return len > 0 && s.charAt(0) > '0';
   }
 
   static double getSimpleNumber(String s) {
```

# Note

```
if (c < '0' || c > '9') {
    return false;
}
```
From the snippet above, we know that `c < 0` is false, so `c != 0` is logically equivalent to `c > 0`