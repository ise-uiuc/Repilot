# Repilot Patch

```
            return 0;
```

# Developer Patch

```
            return 0;
```

# Context

```
--- bug/JxPath-5/src/java/org/apache/commons/jxpath/ri/model/NodePointer.java

+++ fix/JxPath-5/src/java/org/apache/commons/jxpath/ri/model/NodePointer.java

@@ -662,9 +662,7 @@

         }
 
         if (depth1 == 1) {
-            throw new JXPathException(
-                    "Cannot compare pointers that do not belong to the same tree: '"
-                            + p1 + "' and '" + p2 + "'");
+            return 0;
         }
         int r = compareNodePointers(p1.parent, depth1 - 1, p2.parent, depth2 - 1);
         if (r != 0) {
```

# Note

