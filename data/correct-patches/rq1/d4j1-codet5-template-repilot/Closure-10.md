# Repilot Patch

```

      return anyResultsMatch(n, MAY_BE_STRING_PREDICATE);
```

# Developer Patch

```
      return anyResultsMatch(n, MAY_BE_STRING_PREDICATE);
```

# Context

```
--- bug/Closure-10/src/com/google/javascript/jscomp/NodeUtil.java

+++ fix/Closure-10/src/com/google/javascript/jscomp/NodeUtil.java

@@ -1414,7 +1414,8 @@

 
   static boolean mayBeString(Node n, boolean recurse) {
     if (recurse) {
-      return allResultsMatch(n, MAY_BE_STRING_PREDICATE);
+
+      return anyResultsMatch(n, MAY_BE_STRING_PREDICATE);
     } else {
       return mayBeStringHelper(n);
     }
```

# Note

